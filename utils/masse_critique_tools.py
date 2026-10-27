"""
A module to follow the Babelio "Masse Critique" operations.

    1. Calendar: when is the next operation, and what is its theme?
    2. Scraping: the books of the operation currently shown on Babelio, then
       the page of each book (tags, rating, ISBN, series...).
    3. Translation: Babelio is in French, the model works on English text.
       Summaries are translated locally (MarianMT, no API, no call limit); the
       tags are matched to the subjects of the Open Library catalogue; titles
       are kept in French.

Writes book_masse_critique.json, with the same schema as books.json
(English `summary`, catalogue `subjects`) plus the French originals
(`summary_fr`, `subjects_fr`).

Usage (from the project root):
    pip install requests beautifulsoup4 sentencepiece
    python -m utils.masse_critique --out-dir my_infos
    python -m utils.masse_critique --out-dir my_infos --no-details    # skip the book pages
"""


##### IMPORTS ##### ------------------------------------------------------------
import re
import sys
import unicodedata
import json
import time
import hashlib
import argparse
import subprocess
import tempfile
import random as rd
from collections import Counter
import datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from utils.handle_db import clean_text
from utils.subjects import subject_key, canonicalize_subjects
from utils.paths import (CATALOGUE_PATHS, MAPPING_PATH, OVERRIDES_PATH, 
                         VOCAB_CACHE_PATH, PATH_PERSONAL_DATA)
from utils.language import _phrase_is_english

try:
    from zoneinfo import ZoneInfo
    PARIS = ZoneInfo("Europe/Paris")
except Exception:       # no timezone database (e.g. Windows without `tzdata`)
    PARIS = None


##### PARAMETERS ##### ---------------------------------------------------------
URL = "https://www.babelio.com/massecritique.php"
DATASET_NAME = "book_masse_critique.json"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.5",
    "Referer": "https://www.babelio.com/",
}
# If Babelio still answers 403, save the page from your browser (Ctrl+S, "HTML only") here:
LOCAL_PAGE = PATH_PERSONAL_DATA / "masse_critique_page.html"
# The session of a browser that passed Babelio's human check (cookie + user agent), pasted by the
# user in the Masse Critique page. Requests then look like that browser's.
SESSION_PATH = PATH_PERSONAL_DATA / "babelio_session.json"
MAX_DESC = 999          # same truncation as --max-desc in 0-make_book_json.py
MAX_TAGS = 40           # same limit as clean_subjects in handle_db.py
OPENING_HOUR = 18       # applications open at 18:00 (Paris time)
DETAIL_DELAY = 1.0      # seconds between two requests to Babelio's book pages
TRANSLATION_WORKERS = 6 # parallel translation requests (lower it if Google starts refusing them)
TAG_CHUNK_CHARS = 4500  # tags are translated in bulk, under the 5000-character limit of one request

# Where the catalogue's subjects come from (same files as utils.data) and where the tag matches are kept
TAG_THRESHOLD = 0.65    # minimum similarity to keep a tag matched by meaning

# Babelio tags are free user tags: some only describe a reader's shelf, not the book.
NOISE_TAGS = {"livre papier", "presse", "librairie", "pal", "ebook", "kindle",
              "livre numérique", "bibliothèque", "biblio", "à lire", "lu"}
TAG_NOISE_RE = re.compile(r"^(tome|t\.?|vol\.?|volume)\s*\d+$|^\d+$", re.I)

# What a previous run found on the book pages (kept, so that a book is fetched only once)
DETAIL_FIELDS = ("tags_fr", "n_ratings", "avg_rating", "pages",
                 "publication_date", "series", "translators", "age_min", "cover_url",
                 "identifiers", "details_fetched")

# month -> (theme, details). Source: https://www.babelio.com/article/1542/
# Each theme runs twice a year; there is no operation in July and August.
THEMES = {
    1:  ("Littératures", "romans, littérature générale"),
    2:  ("Non-fiction", "essais, documents, sciences humaines"),
    3:  ("Mauvais genres", "polar, thriller, SF, fantasy"),
    4:  ("Graphique", "BD, beaux-livres, illustrés"),
    5:  ("Jeunesse et Jeune adulte", ""),
    6:  ("Non-fiction", "essais, documents, sciences humaines"),
    9:  ("Littératures", "romans, littérature générale"),
    10: ("Mauvais genres", "polar, thriller, SF, fantasy"),
    11: ("Jeunesse et Jeune adulte", ""),
    12: ("Graphique", "BD, beaux-livres, illustrés"),
}


# For auto info scrapping
BOOK_HREF = re.compile(r"/livres/[^/?#]+/(\d+)")
PAGES_RE = re.compile(r"^(\d[\d\s.]*)\s*pages?$", re.I)
AGE_RE = re.compile(r"^[àa] partir de (\d+) ans$", re.I)
CLOSED_RE = re.compile(r"nouvelle op[ée]ration", re.I)      # "A bientôt pour une nouvelle opération !"
REQUESTS_RE = re.compile(r"(\d[\d\s]*)\s+demandes?", re.I)  # "12 demandes" (open operations)
TAG_RE = re.compile(r"^\[([^\]]{2,60})\]\s*")               # "[LIVRE RESERVE A UN PUBLIC AVERTI]"
_UTF8_SEQ = re.compile(rb"[\xc2-\xdf][\x80-\xbf]|[\xe0-\xef][\x80-\xbf]{2}|[\xf0-\xf4][\x80-\xbf]{3}")
CAPTCHA_MARKERS = ("slider-track", "rcaptcha.js")


##### CALENDAR ##### -----------------------------------------------------------
def second_monday(year, month):
    """
        Determines the date (day) of the 2nd monday of a month
    
        Parameters
        ----------
        year: int
        month: int

        Returns
        -------
        The dt.date of the 2nd Monday of a month.
    """
    first = dt.date(year, month, 1)
    first_monday = first + dt.timedelta(days=(-first.weekday()) % 7)    # Monday = 0
    return first_monday + dt.timedelta(days=7)

def upcoming_events(n=1, now=None):
    """
        The next `n` Masse Critique operations that have not opened yet.

        Parameters
        ----------
        n : int
            Number of operations to return.
        now : datetime or None
            Reference time (default: now, Paris time).

        Returns
        -------
        A list of dicts {"opens": datetime, "theme": str, "details": str}.
    """
    now = now or (dt.datetime.now(PARIS) if PARIS else dt.datetime.now())
    events, year, month = [], now.year, now.month
    while len(events) < n:
        if month in THEMES:
            opens = dt.datetime.combine(
                second_monday(year, month), dt.time(OPENING_HOUR), tzinfo=now.tzinfo)
            if opens > now:
                theme, details = THEMES[month]
                events.append({"opens": opens, "theme": theme, "details": details})
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return events


##### SCRAPING: LIST PAGE ##### ------------------------------------------------
def decode_mixed(raw: bytes):
    """
        Decodes a page whose encoding is mixed: Babelio serves ISO-8859-1 /
        cp1252, but some publisher texts are pasted in UTF-8. Valid UTF-8
        sequences are decoded as UTF-8, any other byte as cp1252.

        Parameters
        ----------
        raw: bytes

        Returns
        -------
        A string
    """
    out, pos = [], 0
    for m in _UTF8_SEQ.finditer(raw):
        out.append(raw[pos:m.start()].decode("cp1252", errors="replace"))
        out.append(m.group().decode("utf-8", errors="replace"))
        pos = m.end()
    out.append(raw[pos:].decode("cp1252", errors="replace"))
    return "".join(out)

class BlockedError(RuntimeError):
    """Babelio answered with its "Vérification de sécurité" slider page instead of the real page."""

def check_not_blocked(html):
    """
        Checks if there is a CAPTCHA blocking Babelio's page.

        Parameters
        ----------
        html: str
            The source code of the page.

        Returns
        -------
        Raises BlockedError if `html` is the human-verification page. 
        Returns `html` otherwise.
    """
    if any(m in html for m in CAPTCHA_MARKERS):
        raise BlockedError("Babelio is asking for a human verification (slider page): the saved session may have expired.")
    return html

def load_session(path=SESSION_PATH):
    """
        Loads the JSON file containing the cookie of the user info.

        Parameters
        ----------
        path: Path
            Path to a JSON file (must contain fields "cookie" and "user_agent")

        Returns
        -------
        The saved browser session: {"cookie": str, "user_agent": str} ({} if none).
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}

def save_session(cookie, user_agent, path=SESSION_PATH):
    """
        Saves the cookie / user agent copied from the browser 
        ("Cookie: " prefixes are removed).

        Parameters
        ----------
        cookie: str
            The string given by the user as the "cookie".
        user_agent: str
            The string given by the user as the "user agent".
        path: PATH
            A JSON file path where these infos will be saved to.
    """
    clean = lambda v: re.sub(r"^\s*(cookie|user-agent)\s*:\s*", "", v or "", flags=re.I).strip()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"cookie": clean(cookie), "user_agent": clean(user_agent)}), encoding="utf-8")

def request_headers():
    """
        Sets the headers to fetch a web page.
        Uses default headers or those from a previous session saved by a user.
    """
    headers, saved = dict(HEADERS), load_session()
    if saved.get("cookie"):
        headers["Cookie"] = saved["cookie"]
    if saved.get("user_agent"):
        headers["User-Agent"] = saved["user_agent"]
    return headers

def fetch_html(url=URL, timeout=30, session=None):
    """
        Fetches the source code at a given URL

        Parameters
        ----------
        url: str
            The url to a web page.
        timeout: int
            How many seconds to wait for the server to send data.
        session: requests.Session object

        Returns
        -------
        str (the source code of the page)
    """
    r = (session or requests).get(url, headers=request_headers(), timeout=timeout)
    r.raise_for_status()
    return check_not_blocked(decode_mixed(r.content))

def read_local_page(path=LOCAL_PAGE):
    """
        A Masse Critique page saved by hand from a browser
        (Ultimate fallback when Babelio blocks the script).

        Parameters
        ----------
        path: Path
            The path to a local html file.

        Returns
        -------
        str (the source code of the page)
    """
    return check_not_blocked(decode_mixed(Path(path).read_bytes()))

def _retry(fn, *args, retries=3, **kwargs):
    """
        A retry wrapper. It runs another function, if it fails it retries a
        specified number of times.

        Parameters
        ----------
        fn: a function.
        *args: the arguments to pass to fn().
        retries: int
            The number of times to retry the function.
    """
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except BlockedError:
            raise
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1.5 * (i + 1))

def _clean(text):
    """
        Adds accents back to their letters from its given text.
        Example: ("e" + combining accent -> "é").
    """
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", text or "")).strip()

def _fix_mojibake(text):
    """
        Partially fixes mojibake: some publisher texts were double-encoded 
        (example: "â€“" instead of "–"), we undo it when possible.
    """
    if any(m in text for m in ("â€", "Â", "Ã")):
        try:
            return text.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return text

def _book_ids(el):
    """Extracts Ids of all the Babelio books linked inside an element (str)."""
    return {m.group(1) for a in el.select('a[href*="/livres/"]')
            if (m := BOOK_HREF.search(a["href"]))}

def _parse_card(bid, box):
    """
        Extracts the fields of one book card (<li class="mc-book">), 
        or None if it is not a book.

        Parameters
        ----------
        bid: ???
            A book's id, see _book_ids().
        box: str
            The html element containing a book card.

        Returns
        -------
        a dict with fields: ["babelio_id", "url", "title", "authors", "pages",
        "publisher", "age_min", "requests", "open", "tag", "summary_fr",
        "cover_url"]
    """
    anchors = [a for a in box.select('a[href*="/livres/"]')
               if (m := BOOK_HREF.search(a["href"])) and m.group(1) == bid]
    if not anchors:
        return None
    img = box.find("img")

    # Title and authors
    el = box.select_one(".mc-book-title")
    title = _clean(el.get_text(" ")) if el else ""
    if not title:       # fallback: shortest link text, or the cover's alt ("Title par Author")
        texts = [t for t in (_clean(a.get_text()) for a in anchors) if t]
        if texts:
            title = min(texts, key=len)
        elif img is not None and img.get("alt"):
            title = img["alt"].rsplit(" par ", 1)[0].strip()
    authors = list(dict.fromkeys(
        n for n in (_clean(a.get_text()) for a in box.select('a[href*="/auteur/"]')) if n))
    if not title or not authors:
        return None

    # Cover (src can be relative: "/couv/CVT_xxx.jpg")
    cover = None
    if img is not None:
        for attr in ("data-src", "data-original", "src"):
            src = img.get(attr)
            if src and not src.startswith("data:"):
                cover = urljoin(URL, src)
                break

    # Meta line: <span>56 pages</span><span>Jungle</span><span>à partir de 11 ans</span>
    # Every part is optional ("Bug Ego, tome 1" only has a publisher).
    pages = age = n_requests = publisher = None
    for span in box.select(".mc-book-meta span"):
        line = _clean(span.get_text())
        if not line:
            continue
        if pages is None and (m := PAGES_RE.match(line)):
            n = int(re.sub(r"\D", "", m.group(1)))
            pages = n if n > 1 else None            # Babelio sometimes says "1 pages" for "unknown"
        elif m := AGE_RE.match(line):
            age = int(m.group(1))
        elif len(line) < 40 and (m := REQUESTS_RE.search(line)):
            n_requests = int(re.sub(r"\D", "", m.group(1)))
        elif publisher is None:
            publisher = line

    # Summary (may be missing, or empty)
    desc = box.select_one(".mc-desc-text")
    summary = _clean(_fix_mojibake(desc.get_text(" "))) if desc else ""
    tag = None
    if m := TAG_RE.match(summary):
        tag, summary = m.group(1).strip(), summary[m.end():]

    return {
        "babelio_id": bid, "url": urljoin(URL, anchors[0]["href"]),
        "title": title, "authors": authors, "pages": pages,
        "publisher": publisher, "age_min": age, "requests": n_requests,
        "open": True, "tag": tag, "summary_fr": clean_text(summary, MAX_DESC),
        "cover_url": cover,
    }

def parse_books(html):
    """
        Finds every book of the page (one <li class="mc-book"> each).

        Whether applications are open is read once for the whole page, in the
        banner above the list ("A bientôt pour une nouvelle opération !" = closed).
        If the card classes disappear, it falls back on the links
        (/livres/<slug>/<id>): the card is then the largest element containing
        the links of this book only.

        Parameters
        ----------
        html: str
            The source code of the page.
        
        Returns
        -------
        THe books' infos (dict)
    """
    soup = BeautifulSoup(html, "html.parser")
    state = soup.select_one(".mc-state")
    closed = bool(state and CLOSED_RE.search(state.get_text(" ")))

    boxes = {}
    for li in soup.select("li.mc-book"):
        ids = _book_ids(li)
        if len(ids) == 1:
            boxes.setdefault(next(iter(ids)), li)
    if not boxes:
        first = {}
        for a in soup.select('a[href*="/livres/"]'):
            if m := BOOK_HREF.search(a["href"]):
                first.setdefault(m.group(1), a)
        for bid, anchor in first.items():
            box = anchor
            while (box.parent is not None and box.parent.name not in ("body", "html", "[document]")
                   and _book_ids(box.parent) == {bid}):
                box = box.parent
            boxes[bid] = box

    books = []
    for bid, box in boxes.items():
        if book := _parse_card(bid, box):
            book["open"] = not closed
            books.append(book)
    return books


##### SCRAPING: BOOK PAGE ##### ------------------------------------------------
ISBN_RE = re.compile(r"\b(97[89]\d{10}|\d{9}[\dXx])\b")
DATE_RE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")
AGE_DETAIL_RE = re.compile(r"partir de (\d+) ans", re.I)

def _to_int(text):
    """A function that converts a text to int (or None)"""
    try:
        return int(re.sub(r"\D", "", str(text) if text is not None else ""))
    except ValueError:
        return None

def _to_float(value):
    """A function that converts a value (supposed text) to float (or None)"""
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None

def _json_ld_book(soup):
    """
        The schema.org "Book" block of the page (rating, ISBN, pages...), or {}.
    
        Parameters
        ----------
        soup: bs4 object

        Returns
        -------
        A dict
    """
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or script.get_text())
        except (ValueError, TypeError):
            continue
        nodes = data.get("@graph", [data]) if isinstance(data, dict) else data
        for node in nodes if isinstance(nodes, list) else []:
            if isinstance(node, dict) and node.get("@type") == "Book":
                return node
    return {}

def _biblio_rows(soup):
    """
        The "Informations bibliographiques" table: 
        {lowercase label: <td> of the value}.

        Parameters
        ----------
        soup: bs4 object

        Returns
        -------
        A dict
    """
    rows = {}
    for tr in soup.select("table.biblio-table tr"):
        cells = tr.find_all("td")
        if len(cells) >= 2:
            rows[_clean(cells[0].get_text()).casefold()] = cells[1]
    return rows

def _meta(soup, prop):
    """Finds all the metadat with a given property (prop) in a bs4 object."""
    el = soup.find("meta", attrs={"property": prop})
    return el.get("content") if el and el.get("content") else None

def parse_details(html):
    """
        Reads the page of one book (https://www.babelio.com/livres/<slug>/<id>).

        Rating, ISBN and page count come from the schema.org JSON-LD block (the
        displayed score is rounded: 3.7 instead of 3.69), the other fields from
        the visible HTML, with fallbacks.

        Parameters
        ----------
        html: str
            The source code of the page.

        Returns
        -------
        A dict: tags (French themes), avg_rating (out of 5), n_ratings, isbn,
        publication_date (of the French edition, dd/mm/yyyy), series,
        translators, age_min, cover_url, pages. Missing information is None / [].
    """
    soup = BeautifulSoup(html, "html.parser")
    ld, rows = _json_ld_book(soup), _biblio_rows(soup)
    out = {"avg_rating": None, "n_ratings": None, "isbn": None, "publication_date": None,
           "age_min": None, "cover_url": None, "pages": None, "translators": [], "series": []}

    # Themes ("tags"): shown twice on the page (hero + full list), without duplicates here
    tags = []
    for a in soup.select('a[rel="tag"][href*="/livres-/"]'):
        t = _clean(a.get_text())
        if t and t not in tags:
            tags.append(t)
    out["tags"] = tags

    # Rating
    rating = ld.get("aggregateRating") or {}
    out["avg_rating"] = _to_float(rating.get("ratingValue"))
    out["n_ratings"] = _to_int(rating.get("ratingCount"))
    if out["avg_rating"] is None and (el := soup.select_one(".hero-score[aria-label]")):
        if m := re.search(r"\d+(?:[.,]\d+)?", el["aria-label"]):      # "Note 3,69 sur 5"
            out["avg_rating"] = _to_float(m.group())
    if out["avg_rating"] is not None:
        out["avg_rating"] = round(out["avg_rating"], 2)
        if out["n_ratings"] is None and (el := soup.select_one(".hero-score-count strong")):
            out["n_ratings"] = _to_int(el.get_text())

    # ISBN, date, pages
    isbn = ld.get("isbn") or _meta(soup, "book:isbn") or (_clean(rows["ean"].get_text()) if "ean" in rows else "")
    if m := ISBN_RE.search(str(isbn or "")):
        out["isbn"] = m.group(1)
    if m := re.match(r"(\d{4})-(\d{2})-(\d{2})", _meta(soup, "book:release_date") or ""):
        out["publication_date"] = f"{m.group(3)}/{m.group(2)}/{m.group(1)}"
    elif "publication" in rows and (m := DATE_RE.search(rows["publication"].get_text())):
        out["publication_date"] = m.group(1)
    out["pages"] = _to_int(ld.get("numberOfPages")) or (_to_int(rows["pages"].get_text()) if "pages" in rows else None)

    # Series: "Le Brasier · tome 2 sur 2" under the title (nav.hero-serie), repeated in the
    # table ("Série" row). NB: the first /serie/ link of the hero is "Voir les 2 tomes": not the name.
    name = soup.select_one(".hero-serie-name")
    if name and _clean(name.get_text()):
        out["series"] = [_clean(name.get_text())]
    elif "série" in rows and (a := rows["série"].select_one('a[href*="/serie/"]')) and _clean(a.get_text()):
        out["series"] = [_clean(a.get_text())]

    # Translators: a "(Traducteur)" right after an author's name, or a "Traducteur" row of the table
    for a in soup.select("a.book-author-link, table.biblio-table a.biblio-link"):
        tail = a.next_sibling if isinstance(a.next_sibling, str) else ""
        if re.search(r"traduct", tail, re.I) and _clean(a.get_text()) not in out["translators"]:
            out["translators"].append(_clean(a.get_text()))
    for label, cell in rows.items():
        if label.startswith("traduct"):
            out["translators"] += [n for n in (_clean(a.get_text()) for a in cell.find_all("a"))
                                   if n and n not in out["translators"]]

    # Minimum reading age
    for a in soup.select('a[href$="/age_lecture"]'):
        if m := AGE_DETAIL_RE.search(a.get_text()):
            out["age_min"] = int(m.group(1))
            break

    # Cover
    img = soup.select_one("img#main-cover, img.book-cover")
    src = (img.get("src") if img is not None else None) or _meta(soup, "og:image")
    if src and not src.startswith("data:"):
        out["cover_url"] = urljoin(URL, src)
    return out


##### DATASET ##### ------------------------------------------------------------
def to_record(raw):
    """
        Converts a scraped book into the books.json schema 
        (English fields are filled later).

        Parameters
        ----------
        soup: bs4 object

        Returns
        -------
        A dict
    """
    return {
        "id": f"MC-{raw['babelio_id']}",
        "babelio_id": raw["babelio_id"],
        "url": raw["url"],
        "title": raw["title"],                      # kept in French
        "subtitle": None,
        "author": ", ".join(raw["authors"][:3]) or "Unknown",
        "authors": [{"name": a, "personal_name": None, "alternate_names": []} for a in raw["authors"]],
        "subjects": [],                             # catalogue subjects matched by the tags, see map_tags
        "subjects_map": [],
        "tags_fr": [],                              # all the Babelio tags, before matching
        "summary": None,                            # English, see translate_books
        "summary_fr": raw["summary_fr"],
        "description_source": "babelio_masse_critique",
        "first_sentence": None,
        "notes": None,
        "tag": raw["tag"],
        "series": [],
        "translators": [],
        "publishers": [raw["publisher"]] if raw["publisher"] else [],
        "first_publish_date": None,                 # Babelio only gives the date of the French edition
        "first_publish_year": None,
        "publication_date": None,
        "pages": raw["pages"],
        "age_min": raw["age_min"],
        "covers": [],
        "cover_url": raw["cover_url"],
        "identifiers": {"babelio": [raw["babelio_id"]]},
        "n_english_editions": 0,
        "n_ratings": 0,
        "avg_rating": None,
        "requests": raw["requests"],
        "operation_open": raw["open"],
        "popularity": raw["requests"] or 0,
        "details_fetched": False,
    }

def _set_popularity(record):
    """Sets popularity to available ratings or requests or None."""
    record["popularity"] = record.get("n_ratings") or record.get("requests") or 0

def _apply_details(record, d):
    """Renames some fields in record from d (both are dict)"""
    record["tags_fr"] = d["tags"]
    record["n_ratings"] = d["n_ratings"] or 0
    record["avg_rating"] = d["avg_rating"]
    record["publication_date"] = d["publication_date"]
    record["translators"] = d["translators"]
    if d["series"]:
        record["series"] = d["series"]
    if d["isbn"]:
        record["identifiers"]["isbn"] = [d["isbn"]]
    record["age_min"] = d["age_min"] or record["age_min"]
    record["pages"] = d["pages"] or record["pages"]
    record["cover_url"] = d["cover_url"] or record["cover_url"]
    record["details_fetched"] = True
    _set_popularity(record)

def load_dataset(path: Path):
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


##### LOCAL MODELS ##### -------------------------------------------------------
# Translation and tag matching run in a separate process (utils/text_worker.py),
# like the embeddings: torch and transformers never load in the Streamlit process.
def _run_worker(task, payload, progress=None, label="", module="utils.text_worker"):
    """
        Runs `python -m <module> <task> in.json out.json` and returns the parsed
        output. The worker prints LOADING / PROGRESS <done> <total> / DONE.

        Raises
        ------
        RuntimeError (with the end of the worker's error output) if it fails.
    """
    root = Path(__file__).resolve().parent.parent       # project folder: the worker imports `utils`
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, err = Path(tmp) / "in.json", Path(tmp) / "out.json", Path(tmp) / "stderr.txt"
        src.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with open(err, "w", encoding="utf-8") as err_file:
            proc = subprocess.Popen(
                [sys.executable, "-m", module, task, str(src), str(dst)],
                cwd=root, stdout=subprocess.PIPE, stderr=err_file, text=True)
            for line in proc.stdout:
                parts = line.split()
                if not progress:
                    continue
                if parts[:1] == ["LOADING"]:
                    progress(0, 1, f"loading the model for {label} (the first run downloads it)")
                elif parts[:1] == ["PROGRESS"] and len(parts) == 3:
                    progress(int(parts[1]), int(parts[2]), label)
            proc.wait()
        if proc.returncode != 0 or not dst.exists():
            details = err.read_text(encoding="utf-8", errors="replace").strip()
            hint = " (pip install sentencepiece?)" if "sentencepiece" in details.lower() else ""
            # the useful part is the last "XxxError: ..." line, not the end of a long list
            errors = [l for l in details.splitlines() if re.match(r"\s*[\w.]*(Error|Exception)\b", l)]
            summary = errors[-1][:600] if errors else "..." + details[-400:]
            raise RuntimeError(f"the {task} process failed{hint}: {summary}")
        return json.loads(dst.read_text(encoding="utf-8"))

def translate_books(records, previous=(), progress=None):
    """
        Fills `summary` (English) of each record, in place, with a local
        MarianMT model. A book already translated in `previous` (same French
        summary) is not translated again.

        Parameters
        ----------
        records : list[dict]
        previous : list[dict]
            The previous content of the dataset (used as a translation cache).
        progress : callable(done, total, label) or None

        Returns
        -------
        The ids of the books whose translation came out empty.

        Raises
        ------
        RuntimeError if the translation process fails.
    """
    cache = {r["id"]: r for r in previous}
    todo = []
    for r in records:
        old = cache.get(r["id"])
        if old and old.get("summary") and old.get("summary_fr") == r["summary_fr"]:
            r["summary"] = old["summary"]
        elif not r["summary_fr"]:
            r["summary"] = ""
        else:
            todo.append(r)
    if not todo:
        return []
    out = _run_worker("translate", {"texts": [r["summary_fr"] for r in todo]}, progress, "translating summaries")
    for r, text in zip(todo, out["texts"]):
        r["summary"] = text.strip() or None
    return [r["id"] for r in todo if not r["summary"]]

def enrich_books(records, progress=None, delay=DETAIL_DELAY):
    """
        Reads the Babelio page of each book that has not been read yet and
        fills tags, rating, ISBN, series... in place. One request per book,
        `delay` seconds apart.

        Returns
        -------
        The list of the ids whose page could not be read.
    """
    todo = [r for r in records if not r.get("details_fetched") and r.get("url")]
    failed = []
    with requests.Session() as session:
        for i, r in enumerate(todo, 1):
            try:
                _apply_details(r, parse_details(_retry(fetch_html, r["url"], session=session)))
            except BlockedError:
                failed += [x["id"] for x in todo[i - 1:]]
                print("[WARNING]: Babelio asks for a human verification: the book pages are skipped "
                      "(they will be read next time).")
                break
            except Exception as exc:
                failed.append(r["id"])
                print(f"[WARNING]: could not read the page of {r['id']} ({exc})")
            if progress:
                progress(i, len(todo), "reading book pages")
            time.sleep(delay * (1 + rd.random()))
    return failed


##### TAGS ##### ---------------------------------------------------------------
# Babelio tags are matched to the subjects of the Open Library catalogue (the
# only ones the model knows), instead of being translated:
#   1. tag_overrides.json : your own choices, always win   {"polar": "Detective and mystery stories", "pal": null}
#   2. same subject once normalised ("Science-fiction" = "science fiction")
#   3. closest subject by meaning, with a multilingual sentence model
# Matches 1 and 2 are certain; 3 is kept only above TAG_THRESHOLD (see tag_mapping.json).
def catalogue_subjects(paths=CATALOGUE_PATHS, min_df=5):
    """
        The subjects the model can know: those of at least `min_df` books of the
        catalogue (the vectorizer of utils.recommender uses min_df=2), written
        like in the catalogue (utils.data.load_books merges the variants too).

        Parameters
        ----------
        paths: tuple or list
            A set of paths to books.json-like files.
        min_df: int
            The minimal number of subjects already existing in "paths".

        Returns
        -------
        A dict subject -> number of books.
    """
    books = []
    for path in paths:
        if Path(path).exists():
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            books.extend(data if isinstance(data, list) else [])
    canonicalize_subjects(books)
    count, forms = Counter(), {}
    for b in books:
        for s in b.get("subjects") or []:
            key = subject_key(s)
            if key:
                count[key] += 1
                forms.setdefault(key, s)
    return {forms[k]: n for k, n in count.items()
            if n >= min_df and len(forms[k]) >= 4 and _phrase_is_english(forms[k])}

def _read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default

def _apply_subjects(record, lookup):
    """subjects = the catalogue subjects matched by the tags, without the shelf tags and the duplicates."""
    authors = {a["name"].casefold() for a in record.get("authors") or []}
    seen, subjects, mapping = set(), [], {}
    for fr in record.get("tags_fr") or []:
        if fr.casefold() in NOISE_TAGS or fr.casefold() in authors or TAG_NOISE_RE.match(fr):
            continue
        s = lookup(fr)
        if not s or subject_key(s) in seen:
            continue
        seen.add(subject_key(s))
        subjects.append(s)
        mapping[fr] = s
    record["subjects"] = subjects[:MAX_TAGS]
    record["subjects_map"] = dict(list(mapping.items())[:MAX_TAGS])

def map_tags(records, progress=None, threshold=TAG_THRESHOLD, catalogue=CATALOGUE_PATHS,
             mapping_path=MAPPING_PATH, overrides_path=OVERRIDES_PATH, vocab_cache=VOCAB_CACHE_PATH):
    """
        Fills `subjects` (catalogue subjects) and `subjects_fr` (the tags they
        come from) of each record. Each tag is matched once: results are kept
        in `mapping_path` (reviewable) until the catalogue's subjects change.
        A failing embedding model is not fatal: the tags it should have matched
        are simply left out, and tried again next time.
    """
    vocab = sorted(catalogue_subjects(catalogue))
    if not vocab:
        print(f"[WARNING]: no subject found in {', '.join(map(str, catalogue))}: tags are not matched.")
        return
    keys = {subject_key(s): s for s in vocab}
    vocab_hash = hashlib.md5("\n".join(vocab).encode()).hexdigest()[:12]
    stored = _read_json(mapping_path, {})
    tags = stored.get("tags", {}) if stored.get("vocab_hash") == vocab_hash else {}
    overrides = {k.casefold(): v for k, v in _read_json(overrides_path, {}).items()}

    wanted = sorted({t for r in records for t in r.get("tags_fr") or []
                     if t.casefold() not in NOISE_TAGS and t.casefold() not in overrides
                     and t not in tags and not TAG_NOISE_RE.match(t)})
    todo = []
    for t in wanted:
        if (k := subject_key(t)) in keys:
            tags[t] = {"subject": keys[k], "score": 1.0, "source": "exact"}
        elif len(t) < 4:        # too short for a reliable embedding match: use tag_overrides.json
            continue
        else:
            todo.append(t)
    if todo:
        try:
            out = _run_worker("match", {"queries": todo, "vocab": vocab, "cache": str(vocab_cache)},
                              progress, "matching tags")
            for t, (j, score) in zip(todo, out["best"]):
                tags[t] = {"subject": vocab[j], "score": round(score, 3), "source": "embedding"}
        except RuntimeError as exc:
            print(f"[WARNING]: {len(todo)} tags could not be matched ({exc})")
    Path(mapping_path).parent.mkdir(parents=True, exist_ok=True)
    Path(mapping_path).write_text(json.dumps(
        {"vocab_hash": vocab_hash, "tags": dict(sorted(tags.items()))}, ensure_ascii=False, indent=1), encoding="utf-8")

    def lookup(tag):
        if tag.casefold() in overrides:
            return overrides[tag.casefold()]
        m = tags.get(tag)
        return m["subject"] if m and m["score"] >= threshold else None

    for r in records:
        _apply_subjects(r, lookup)


##### BUILD ##### --------------------------------------------------------------
def build_dataset(out_path: Path, progress=None, details=True, tag_threshold=TAG_THRESHOLD):
    """
        Scrapes Babelio, translates the summaries, matches the tags and writes
        the JSON file.

        Parameters
        ----------
        out_path : Path
        progress : callable(done, total, label) or None
        details : bool
            Also read the page of each book (slower: one request per new book).
        tag_threshold : float
            Minimum similarity to keep a tag matched by meaning.

        Returns
        -------
        A tuple (records, ids not translated, ids whose page could not be read).
    """
    try:
        html = fetch_html()
    except (requests.HTTPError, BlockedError) as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(exc, requests.HTTPError) and status not in (403, 429):
            raise
        how = "asks for a human verification" if isinstance(exc, BlockedError) else f"refused the request (HTTP {status})"
        if not LOCAL_PAGE.exists():
            raise RuntimeError(
                f"Babelio {how}. Pass the check in your browser, then either paste your browser session in "
                f"the 'Babelio verification' box, or save the page (Ctrl+S, 'HTML only') as {LOCAL_PAGE}.") from exc
        print(f"[WARNING]: Babelio {how}: using the saved page {LOCAL_PAGE}")
        try:
            html = read_local_page()
        except BlockedError as exc2:
            raise RuntimeError(f"{LOCAL_PAGE} is the verification page, not the book list: pass the check "
                               "in your browser first, then save the page again.") from exc2
    raw = parse_books(html)
    if not raw:
        raise RuntimeError("No book found on the page: Babelio's layout may have changed.")
    previous = load_dataset(out_path)
    old_by_id = {r["id"]: r for r in previous}

    records = [to_record(b) for b in raw]
    for r in records:                               # what is already known is not fetched again
        old = old_by_id.get(r["id"])
        if old and old.get("details_fetched"):
            r.update({k: old[k] for k in DETAIL_FIELDS if k in old})
            _set_popularity(r)

    failed = translate_books(records, previous, progress)
    failed_details = enrich_books(records, progress) if details else []
    map_tags(records, progress, threshold=tag_threshold)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(out_path)
    return records, failed, failed_details


##### MAIN ##### ---------------------------------------------------------------
def main():
    """The main caller"""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=PATH_PERSONAL_DATA, type=Path)
    ap.add_argument("--no-details", action="store_true", help="do not read the page of each book")
    ap.add_argument("--tag-threshold", default=TAG_THRESHOLD, type=float,
                    help="minimum similarity to keep a tag matched by meaning")
    args = ap.parse_args()

    def progress(done, total, label):
        print(f"\r  {label}: {done:,}/{total:,}", end="\n" if done >= total else "", flush=True)

    out_path = args.out_dir / DATASET_NAME
    print(f"Fetching {URL} ...")
    records, failed, failed_details = build_dataset(
        out_path, progress, details=not args.no_details, tag_threshold=args.tag_threshold)
    state = "open" if any(r["operation_open"] for r in records) else "closed"
    print(f"\n✓ Wrote {len(records)} books to {out_path} (applications {state}, "
          f"{len(failed)} not translated, {len(failed_details)} pages not read)")


if __name__ == "__main__":
    main()
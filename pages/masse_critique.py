"""
    Masse Critique page (shown through app.py): streamlit run app.py
    1. Shows the next Masse Critique operation (2nd Monday of the month, rotating theme)
    2. Fetches the books of the current operation (book_masse_critique.json)
       with summaries translated to English by a local model and tags matched to the catalogue's subjects
       (titles stay in French)
    3. Scores them with the model saved by the Recommendation page (model/)
    4. Shows the N most interesting ones
"""

##### IMPORTS ##### ------------------------------------------------------------
import json
import hashlib
import datetime as dt
from collections import Counter
from pathlib import Path

import streamlit as st

from utils.paths import (PATH_PERSONAL_DATA, PATH_MC_EMBEDDINGS, LABELS_PATH,
                         MAPPING_PATH, OVERRIDES_PATH, MODEL_DIR)
from utils.data import TARGET, read_labels
from utils.embeddings import EmbeddingsUnavailable
from utils.masse_critique_tools import (
    DATASET_NAME, TAG_THRESHOLD, build_dataset, load_session, save_session, upcoming_events)
from utils.cover import cover


##### PARAMETERS ##### ---------------------------------------------------------
BOOKS_PATH = PATH_PERSONAL_DATA / DATASET_NAME
MODEL_FILE = MODEL_DIR / "model.joblib"     # written by the Recommendation page
EMB_CACHE = PATH_MC_EMBEDDINGS              # NOT embeddings.npz
DEFAULT_SHOWN = 10                          # Number of recommendations to show

ss = st.session_state


##### FUNCTIONS ##### ----------------------------------------------------------
def mtime(path):
    """
        Helper that shows the datetime of a local Path (file or folder).

        Parameters
        ----------
        path: Path
        
        Returns
        -------
        Integer 
    """
    return path.stat().st_mtime if path.exists() else 0

def labels_digest(rows):
    """ 
        Generates a fingerprint from book infos
        (same fingerprint as save_result() in utils.recommender).

        Parameters
        ----------
        rows: dict
            Contains fields "book_id", "label" and "read"
        
        Returns
        -------
        hashlib.md5 fingerprints
    """
    return hashlib.md5(json.dumps(sorted(
        (r["book_id"], str(r["label"]), str(r["read"])) for r in rows)).encode()).hexdigest()[:12]

def read_json(path):
    """A helper to read a JSON file. Parameter: Path."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

def read_metrics():
    """Helper to read the metrics of the model from file."""
    return read_json(MODEL_DIR / "metrics.json")

@st.cache_data
def load_books():
    """A Helper to read the books metadata in JSON file. """
    with open(BOOKS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []

@st.cache_data(show_spinner="Scoring books...")
def rank_books():
    """
        A function to rank books.

        Returns
        -------
            A tuple 
                - ([(book, proba, reasons)] by decreasing probability, 
                - (known, total) tags)
            or None if there is no saved model.
    """
    if not MODEL_FILE.exists():
        return None
    from utils.recommender import score_new_books           # imports scikit-learn: only when needed
    books = {b["id"]: b for b in load_books() if b.get("summary")}
    scores = score_new_books(books, MODEL_DIR, cache_path=EMB_CACHE)
    order = sorted(books, key=lambda i: scores[i]["proba"], reverse=True)
    ranked = [(books[i], scores[i]["proba"], scores[i]["reasons"]) for i in order]
    known = (sum(s["subjects_known"] for s in scores.values()), sum(s["subjects_total"] for s in scores.values()))
    return ranked, known

def run_fetch(details):
    """
        A helper to fetch books and show error if necessary.

        Parameters
        ----------
        details: bool
            If True, reads each book's individual page as well as its summary
            on masse critique homepage.

        Returns
        -------
        The outputs of build_dataset.
    """
    bar = st.progress(0.0, text="Fetching Babelio...")

    def progress(done, total, label):
        bar.progress(min(done / max(total, 1), 1.0), text=f"{label.capitalize()}: {done:,}/{total:,}")

    try:
        records, failed, failed_details = build_dataset(BOOKS_PATH, progress, details=details)
    except Exception as exc:
        bar.empty()
        st.error(f"Could not fetch the books: {exc}")
        return
    bar.empty()
    problems = [f"{len(failed)} not translated" if failed else "",
                f"{len(failed_details)} pages not read" if failed_details else ""]
    ss["mc_flash"] = f"{len(records)} books fetched" + "".join(f", {p}" for p in problems if p)
    st.rerun()

def render_book(book, rank, proba, reasons):
    """
        Shows the info card of each book.

        Parameters
        ----------
        book: dict
            A book's infos.
        rank: int
            It's rank according to the model.
        proba: float
            The confidence of the model.
        reasons: list
            A list of "reasons" (usually tags) for selecting this book.
    """
    with st.container(border=True):
        left, right = st.columns([1, 5], vertical_alignment="center")
        with left:
            cover(book.get("cover_url"), book["title"])
        with right:
            st.markdown(f"#### {rank}. {book['title']}")
            meta = [book.get("author") or "Unknown author"]
            if book.get("publishers"):
                meta.append(book["publishers"][0])
            if book.get("pages"):
                meta.append(f"{book['pages']} pages")
            if book.get("age_min"):
                meta.append(f"{book['age_min']}+")
            st.write(" · ".join(meta))
            if book.get("avg_rating"):
                st.caption(f":material/star: {book['avg_rating']} avg · {book.get('n_ratings', 0):,} ratings on Babelio")
            if book.get("subjects_map"):
                st.write(" ".join(f"`{fr}`" for fr, en in list(book["subjects_map"].items())[:8]))
            if book.get("tag"):
                st.caption(f":material/warning: {book['tag'].capitalize()}")
            if proba is not None:
                st.progress(proba, text=f"Confidence: {proba:.0%} chance you'll like it")
            if reasons:
                st.caption("Why: " + " · ".join(f"`{r}`" for r in reasons))
            with st.expander("Summary"):
                st.write(book["summary"])
            with st.expander("Résumé (French)"):
                st.write(book["summary_fr"])
            if book.get("url"):
                st.link_button("Open on Babelio", book["url"], icon=":material/open_in_new:")


##### PAGE ##### ---------------------------------------------------------------
if ss.get("mc_flash"):
    st.toast(ss.pop("mc_flash"))

st.title("Masse Critique")

# 1. Show next Masse Critique operation
events = upcoming_events(6)
nxt = events[0]
days = (nxt["opens"].date() - dt.datetime.now(nxt["opens"].tzinfo).date()).days
st.subheader("Next operation")
c1, c2, c3 = st.columns(3)
c1.metric("Opens", f"{nxt['opens']:%a %d %b %Y} · {nxt['opens']:%H:%M}")
c2.metric("In", f"{days} day{'s' if days != 1 else ''}")
c3.metric("Theme", nxt["theme"], help=nxt["details"] or None)
st.caption("Operations start on the 2nd Monday of each month at 18:00 (Paris time), except in July and "
           "August. Invitation-only operations are not listed.")
with st.expander("Upcoming schedule"):
    st.dataframe(
        [{"Opens": f"{e['opens']:%a %d %b %Y}", "Theme": e["theme"], "Details": e["details"]} for e in events],
        hide_index=True, width="stretch")

with st.expander("Babelio verification (if fetching failed)", expanded=not load_session()):
    st.markdown(
        "Babelio sometimes shows a *\"Vérification de sécurité\"* slider page. "
        "This is to prevent bots. Fortunately, you are a human, not a bot, and "
        "you can help to overcome this:\n\n"
        "1. Open [Masse Critique](https://www.babelio.com/massecritique.php) "
        "in your browser and pass the check.\n"
        "2. Press **F12**, open the **Network** tab, reload the page and click "
        "the first request (`massecritique.php`).\n"
        "3. In *Request Headers*, copy the value of **cookie** and of "
        "**user-agent** below.\n\n"
        "This will allow the app to fetch books at a slow pace (one page per "
        "second) to avoid overloading the website.")
    c_cookie = st.text_input("cookie", type="password", placeholder="paste the value of the cookie header")
    c_agent = st.text_input("user-agent", placeholder="Mozilla/5.0 (...)")
    if st.button("Save session", icon=":material/key:", disabled=not (c_cookie and c_agent)):
        save_session(c_cookie, c_agent)
        st.toast("Session saved: fetch again.")
        st.rerun()
    st.caption("A session is saved." if load_session() else "No session saved.")
    st.caption("It expires after a while: if the check comes back, repeat these steps. "
               "It is stored in `my_infos/babelio_session.json` on your computer only.")
info, option, action = st.columns([3, 2, 1], vertical_alignment="center")
has_file = BOOKS_PATH.exists()
if has_file:
    info.caption(f"Last fetched: {dt.datetime.fromtimestamp(mtime(BOOKS_PATH)):%Y-%m-%d %H:%M}")
else:
    info.info("No book fetched yet.")
fetch_details = option.toggle(
    "Read each book's page", value=True,
    help="Adds tags, rating, ISBN, series. One request per new book (about 1 s each); "
         "books already read are not fetched again. Translation runs locally: the first run "
         "downloads the models.")
if action.button("Fetch from Babelio", icon=":material/sync:", width="stretch",
                 type="secondary" if has_file else "primary"):
    run_fetch(fetch_details)
if not has_file:
    st.stop()

books = load_books()
if not books:
    st.warning("The file is empty. Try fetching again.")
    st.stop()
if any(b.get("operation_open") for b in books):
    st.success("Applications are open for these books.", icon=":material/lock_open:")
else:
    st.warning("Applications are closed: Babelio still shows the books of the last operation. "
               f"They will be replaced when the next one opens ({nxt['opens']:%d %b}); fetch again then.",
               icon=":material/lock:")
n_missing = sum(1 for b in books if not b.get("summary"))
if n_missing:
    st.caption(f"{n_missing} book(s) without an English summary are left out. Fetch again to retry the translation.")
books = [b for b in books if b.get("summary")]

# 2. Ranking of recommended books
head, field = st.columns([4, 1], vertical_alignment="bottom")
head.subheader("Ranked by interest")
n_show = field.number_input("Books to show", min_value=1, max_value=max(len(books), 1),
                            value=min(DEFAULT_SHOWN, max(len(books), 1)), step=1)

labels = read_labels()
metrics = read_metrics()
ranked = known = None
if not MODEL_FILE.exists():
    st.info(f"The recommendation model isn't trained yet ({len(labels)}/{TARGET} labels). Rate books on the "
            "Annotation page, then open the Recommendation page once: it trains and saves the model. "
            "Until then these books are shown in Babelio's order.", icon=":material/model_training:")
else:
    try:
        ranked, known = rank_books() or (None, None)
    except (EmbeddingsUnavailable, ValueError) as exc:
        st.error(f"Could not score the books: {exc}", icon=":material/error:")
    if metrics and metrics.get("labels_hash") != labels_digest(labels):
        st.warning(f"The model was trained on {metrics['n_labels']} labels, but your labels changed since: "
                   "open the Recommendation page to retrain it.", icon=":material/update:")
    elif metrics:
        st.caption(f"Model trained on {metrics['n_labels']} labels ({metrics['trained_at']}).")

if known and known[1]:
    st.caption(f"{known[0]} of {known[1]} Babelio tags match subjects the model knows; the others are ignored "
               "(see SUBJECT_ALIASES in utils/recommender.py to map more).")

shown = ranked if ranked else [(b, None, []) for b in books]
st.caption(f"Showing {min(n_show, len(shown))} of {len(shown)} books"
           + (", best first." if ranked else "."))

# 3. Display
for i, (book, proba, reasons) in enumerate(shown[:n_show], 1):
    render_book(book, i, proba, reasons)

# 4. Tag matching, to check
mapping = (read_json(MAPPING_PATH) or {}).get("tags", {})
if mapping:
    overrides = {k.casefold(): v for k, v in (read_json(OVERRIDES_PATH) or {}).items()}
    freq = Counter(t for b in books for t in b.get("tags_fr") or [])
    rows = []
    for tag, n in freq.most_common(60):
        m = mapping.get(tag)
        if tag.casefold() in overrides:
            subject, score, source = overrides[tag.casefold()], None, "override"
        elif m:
            subject, score, source = m["subject"], m["score"], m["source"]
            if score < TAG_THRESHOLD:
                subject, source = None, f"{source} (below {TAG_THRESHOLD})"
        else:
            continue
        rows.append({"Babelio tag": tag, "Books": n, "Catalogue subject": subject, "Similarity": score, "Source": source})
    with st.expander("How the Babelio tags were matched to the catalogue's subjects"):
        st.caption("The model only knows the subjects of the Open Library catalogue, so each Babelio tag is matched "
                   f"to one of them. To correct a match, add it to `{OVERRIDES_PATH}` (e.g. "
                   '`{"polar": "Detective and mystery stories", "pal": null}`; `null` drops the tag), '
                   "then fetch again: nothing is translated or downloaded twice.")
        st.dataframe(rows, hide_index=True, width="stretch")
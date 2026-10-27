"""
    A set of function to extract data from txt.gz files.

    Disclaimer : I know very little on SQL, so this file was entirely generated 
    by Claude.ai and I only very lightly modified it. Too much risks.
"""

##### IMPORTS ##### ------------------------------------------------------------
import re
import json
import duckdb
import datetime as dt
from pathlib import Path
from collections import Counter, defaultdict

from utils.progress import progress
from utils.subjects import subject_key
from utils.language import subject_is_english


##### PARAMETERS ##### ---------------------------------------------------------
COLS = "{'type':'VARCHAR','key':'VARCHAR','rev':'VARCHAR','modified':'VARCHAR','data':'VARCHAR'}"
BATCH = 100_000         # rows fetched per batch
PRINT_EVERY = 10        # print progress every N batches
BAD_SUBJECT = (
    "accessible", "protected", "daisy", "large type", "in library", 
    "open library", "reading level", "lending", "nyt:", "overdrive", 
    "internet archive", "long now", "bestseller", "new york best",
    "translations", "language edition")

# Cheap regex (no JSON parsing) to pull the parent work key out of an edition line.
WORK_RE = r'"works":\s*\[\s*\{\s*"key":\s*"(/works/OL\d+W)"'

# The "languages" field of an edition must contain English. (A plain substring
# search for "/languages/eng" also matches e.g. "translated_from": eng, which
# lets foreign translations of English books through.)
LANG_ENG_RE = r'"languages":\s*\[[^\]]*"/languages/eng"'

# Shared field extraction (works + editions). Same columns as before.
_WORK_EDITION_SELECT = """
    SELECT
        type, key,
        -- editions ---------------------------------------------------------
        CASE WHEN type = '/type/edition' THEN json_extract_string(data, '$.works[0].key') END AS work_key,
        CASE WHEN type = '/type/edition' THEN TRY_CAST(json_extract(data, '$.publishers') AS VARCHAR[]) END AS publishers,
        CASE WHEN type = '/type/edition' THEN json_extract_string(data, '$.title') END AS edition_title,
        CASE WHEN type = '/type/edition' THEN json_extract_string(data, '$.publish_date') END AS publish_date,
        CASE WHEN type = '/type/edition' THEN TRY_CAST(json_extract_string(data, '$.number_of_pages') AS INTEGER) END AS pages,
        CASE WHEN type = '/type/edition' THEN txt(data, '$.notes') END AS notes,
        CASE WHEN type = '/type/edition' THEN TRY_CAST(json_extract(data, '$.series') AS VARCHAR[]) END AS series,
        CASE WHEN type = '/type/edition' THEN json_extract(data, '$.identifiers')::VARCHAR END AS identifiers,
        -- works ------------------------------------------------------------
        CASE WHEN type = '/type/work' THEN json_extract_string(data, '$.title') END AS title,
        CASE WHEN type = '/type/work' THEN json_extract_string(data, '$.subtitle') END AS subtitle,
        CASE WHEN type = '/type/work' THEN TRY_CAST(json_extract(data, '$.subjects') AS VARCHAR[]) END AS subjects,
        CASE WHEN type = '/type/work' THEN
            list_transform(TRY_CAST(json_extract(data, '$.authors[*].author.key') AS JSON[]),
                            x -> json_extract_string(x, '$')) END AS author_keys,
        CASE WHEN type = '/type/work' THEN json_extract_string(data, '$.first_publish_date') END AS first_publish_date,
        CASE WHEN type = '/type/work' THEN TRY_CAST(json_extract(data, '$.covers') AS BIGINT[]) END AS covers,
        -- works and editions -----------------------------------------------
        txt(data, '$.description') AS description,
        txt(data, '$.first_sentence') AS first_sentence,
        -- authors (filled by the later INSERT) ----------------------------
        NULL::VARCHAR AS name,
        NULL::VARCHAR AS personal_name,
        NULL::VARCHAR[] AS alternate_names
    FROM raw_sel
"""


##### FUNCTIONS ##### ----------------------------------------------------------
def sql_path(p: Path):
    """
        Formats a file path so it can be safely inserted in a SQL string literal.

        Parameters
        ----------
        p : Path
            The path to format.

        Returns
        ------
        A str: the absolute path with forward slashes, in which every single
        quote is doubled (to avoid breaking the SQL query).
    """
    return p.resolve().as_posix().replace("'", "''")

def build_slim(db_path: Path, dump: Path, ratings: Path, n_candidates: int):
    """
        Builds the "slim" table (same schema as before) WITHOUT parsing the JSON
        of the whole dump. Strategy:

          1. From the (small) ratings dump, pick the `n_candidates` most-rated works.
          2. Scan the all-types dump once, keeping the RAW line only for those
             works and for their English editions (cheap filters: the `key`
             column and a regex; no JSON parsing).
          3. Parse JSON only on that small subset.
          4. Scan the dump a second time to fetch only the authors of those works.

        If `db_path` already contains "slim", it is reused.

        Parameters
        ----------
        db_path: Path
            Path to a duckdb file to be created.
        dump: Path
            Path to a duckdb file containing book's infos (main OLDP dump).
        ratings: Path
            Path to a duckdb file containing book's ratings (ratings from OLDP).
        n_candidates: int
            The number of books to extract from these db.

        Returns
        -------
        duckdb object
    """
    con = duckdb.connect(str(db_path))
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='slim'").fetchone()[0]
    if exists and not con.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name='slim' AND column_name='edition_title'").fetchone()[0]:
        print(f"Cached extract {db_path.name} predates the language fixes: rebuilding it.")
        con.execute("DROP TABLE slim")
        exists = 0
    if exists:
        print(f"Using cached extract {db_path.name} "
              "(delete it manually to force a rebuild).")
        return con

    t_start = dt.datetime.now()
    read_dump = (f"read_csv('{sql_path(dump)}', delim='\\t', header=false, quote='', escape='', "
                 f"columns={COLS}, ignore_errors=true)")
    con.execute("""CREATE OR REPLACE TEMP MACRO txt(d, p) AS
                   COALESCE(json_extract_string(d, p || '.value'), json_extract_string(d, p))""")

    # 1) candidate works, from the ratings dump -------------------------------
    print(f"Step 1/4: picking the {n_candidates:,} most-rated works (ratings dump)")
    with progress("  ratings dump", ratings.stat().st_size):
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE cand AS
            SELECT work_key FROM read_csv('{sql_path(ratings)}', delim='\\t', header=false,
                    quote='', escape='',
                    columns={{'work_key':'VARCHAR','edition_key':'VARCHAR','rating':'DOUBLE','d':'VARCHAR'}},
                    ignore_errors=true)
            GROUP BY work_key
            ORDER BY count(*) DESC, work_key
            LIMIT {int(n_candidates)}
        """)

    # 2) raw lines of candidate works + their English editions ----------------
    print("Step 2/4: scanning the all-types dump for those works and their English editions")
    with progress("  all-types dump", dump.stat().st_size):
        con.execute(f"""
            CREATE OR REPLACE TABLE raw_sel AS
            SELECT type, key, data FROM {read_dump}
            WHERE (type = '/type/work' AND key IN (SELECT work_key FROM cand))
               OR (type = '/type/edition'
                   AND data LIKE '%/languages/eng"%'
                   AND regexp_matches(data, '{LANG_ENG_RE}')
                   AND regexp_extract(data, '{WORK_RE}', 1) IN (SELECT work_key FROM cand))
        """)
    n_raw = con.execute("SELECT count(*) FROM raw_sel").fetchone()[0]
    print(f"  kept {n_raw:,} raw rows")

    # 3) JSON parsing, only on the subset -------------------------------------
    print("Step 3/4: parsing JSON of the selected rows")
    with progress("  parsing"):
        con.execute(f"CREATE OR REPLACE TABLE slim AS {_WORK_EDITION_SELECT}")
        con.execute("DROP TABLE raw_sel")
        con.execute("""
            CREATE OR REPLACE TEMP TABLE akeys AS
            SELECT DISTINCT unnest(author_keys) AS k FROM slim WHERE type = '/type/work'
        """)

    # 4) authors of those works ------------------------------------------------
    print("Step 4/4: scanning the all-types dump again for the authors")
    with progress("  all-types dump", dump.stat().st_size):
        con.execute(f"""
            INSERT INTO slim BY NAME
            SELECT type, key,
                   json_extract_string(data, '$.name') AS name,
                   json_extract_string(data, '$.personal_name') AS personal_name,
                   TRY_CAST(json_extract(data, '$.alternate_names') AS VARCHAR[]) AS alternate_names
            FROM {read_dump}
            WHERE type = '/type/author' AND key IN (SELECT k FROM akeys)
        """)
    print(f"Extract finished in {dt.datetime.now() - t_start}.")
    return con

def year_of(text):
    """
        A function to find a year from a text

        Parameters
        ----------
        text : a str

        Returns
        -------
        The year (dt.datetime) or None
    """
    m = re.search(r"\b(1[4-9]\d{2}|20\d{2})\b", text or "")
    if m:
        y = int(m.group(1))
        if y <= dt.date.today().year + 1:
            return y
    return None

def aggregate_editions(con, work_keys, max_per_work=None):
    """
        Merge the English editions of the selected works into one summary
        per work.

        Parameters
        ----------
        con : duckdb.DuckDBPyConnection
            Connection to the extract database (it must contain the `slim` table).
        work_keys : list[str]
            Open Library work keys to aggregate, e.g. "/works/OL45883W".
        max_per_work : int or None
            If set, only the first `max_per_work` editions of each work are
            read (faster; "n" then counts at most this many editions).

        Returns
        -------
        dict[str, dict]
            Maps each work key to its aggregated data:
                - edition count "n" and a counter of edition "titles",
                - publisher and series counters ("pubs", "series"),
                - lists of "years" and "pages",
                - the longest "notes" and "first_sentence",
                - merged external "ids" (up to 5 per source).
    """
    start = dt.datetime.now()

    # 1) Join once, into a temp table (optionally capped per work)
    con.execute(
        "CREATE OR REPLACE TEMP TABLE top_keys AS SELECT unnest(?) AS work_key",
        [work_keys])
    cap = (f"QUALIFY row_number() OVER (PARTITION BY e.work_key ORDER BY e.key) <= {int(max_per_work)}"
           if max_per_work else "")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE sel_editions AS
        SELECT e.work_key, e.edition_title, e.publishers, e.publish_date, e.pages, e.notes,
               e.series, e.first_sentence, e.identifiers
        FROM slim e
        JOIN top_keys t ON e.work_key = t.work_key
        WHERE e.type = '/type/edition'
        {cap}
    """)
    total = con.execute("SELECT count(*) FROM sel_editions").fetchone()[0]
    n_batches = -(-total // BATCH)          # ceiling division
    print(f"[EDITIONS] {total:,} editions to merge ({n_batches} batches).")

    # 2) Stream the rows and merge them per work
    agg = defaultdict(lambda: {
        "n": 0,
        "titles": Counter(),
        "pubs": Counter(),
        "series": Counter(),
        "years": [],
        "pages": [],
        "notes": "",
        "first_sentence": "",
        "ids": defaultdict(list),
    })

    cur = con.execute("SELECT * FROM sel_editions")
    i = 0
    while batch := cur.fetchmany(BATCH):
        i += 1
        if i % PRINT_EVERY == 0 or i == n_batches:
            print(f"[EDITIONS] batch {i}/{n_batches} "
                  f"({min(i * BATCH, total):,}/{total:,} rows, "
                  f"{dt.datetime.now() - start} elapsed)", flush=True)

        for wk, etitle, pubs, pdate, pages, notes, series, fsent, idents in batch:
            a = agg[wk]
            a["n"] += 1
            if etitle and etitle.strip():
                a["titles"][etitle.strip()] += 1
            a["pubs"].update(p.strip() for p in (pubs or []) if p and p.strip())
            a["series"].update(s.strip() for s in (series or []) if s and s.strip())
            if (y := year_of(pdate)):
                a["years"].append(y)
            if pages and 8 <= pages <= 5000:
                a["pages"].append(pages)
            if notes and len(notes) > len(a["notes"]):
                a["notes"] = notes
            if fsent and len(fsent) > len(a["first_sentence"]):
                a["first_sentence"] = fsent
            if idents:
                try:
                    for name, vals in json.loads(idents).items():
                        if isinstance(vals, list):
                            for v in vals:
                                if str(v) not in a["ids"][name] and len(a["ids"][name]) < 5:
                                    a["ids"][name].append(str(v))
                except (ValueError, AttributeError):
                    pass

    print(f"[EDITIONS] done in {dt.datetime.now() - start} "
          f"({len(agg):,} works with editions).")
    return agg

def clean_text(text, max_chars):
    """
        A function to format text fields

        Parameters
        ----------
        text : str
            The text that will be treated
        max_chars : int
            A maximum number of chars for the text (any text longer that this
            threshold will be cutoff)

        Returns
        -------
        The new text, formatted.
    """
    if not text:
        return ""
    text = text.split("----------")[0]      # drop "Contains:/Source:" footers
    text = re.sub(r"<[^>]*>", " ", text)                    # remove HTML tags (<p>, <i>, <a href=...>)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)    # [label](url) -> label
    text = re.sub(r"\[([^\]]+)\]\[\d+\]", r"\1", text)      # [label][1]   -> label
    text = re.sub(r"https?://\S+", "", text)                # remove URLs
    text = re.sub(r"[*#>]+", "", text)                      # remove markdown symbols
    text = re.sub(r"\s+", " ", text).strip()                # normalize whitespaces 
    if len(text) > max_chars:
        cut = text[:max_chars]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        text = cut[:end + 1] if end > max_chars * 0.5 else cut.rstrip() + "…"
    return text

def clean_subjects(subjects, limit=None, max_len=40, vocab=None):
    """
        A function to make short tags (the raw list is kept separately).

        Parameters
        ----------
        subjects : a list
        limit : int
            The maximum number of tags for a subject
        vocab : set[str] or None
            English words (see utils.language.english_vocabulary), used to
            reject one-word tags in other languages ("Novela").

        Returns
        -------
        The list of tags 
    """
    out, seen = [], set()
    for s in subjects or []:
        s = re.sub(r"\s+", " ", (s or "")).strip().rstrip(".")
        key = s.lower()
        if ((not s) or 
            (len(s) > max_len or ":" in s or subject_key(s) in seen) or 
            (any(b in key for b in BAD_SUBJECT)) or
            (not s.isascii()) or
            (not subject_is_english(s, vocab))):
            continue
        seen.add(subject_key(s))
        out.append(s)
        if limit and len(out) == limit:
            break
    return out
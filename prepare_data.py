#!/usr/bin/env python3

"""
This script builds a JSON containing book data, from books fetched from 
Open Library dumps. It can be run locally, but it needs files that can only
be downloaded from (download from https://openlibrary.org/developers/dumps).

Expected files in to place --data-dir:
    ol_dump_YYYY-MM-DD.txt.gz            all-types dump [required]
    ol_dump_ratings_YYYY-MM-DD.txt.gz    ratings dump   [required]

For basic usage, simply run this in terminal:
    $ python prepare_data.py
"""


##### IMPORTS ##### ------------------------------------------------------------
import re
import sys
import json
import argparse
import statistics
from collections import Counter
from pathlib import Path

from utils.check_files import find_dump, check_age
from utils.subjects import canonicalize_subjects
from utils.progress import progress, track, fraction_progress
from utils.embeddings import embed_books, EmbeddingsUnavailable
from utils.language import keep_english, english_title, english_vocabulary
from utils.handle_db import (sql_path, build_slim, aggregate_editions, 
                             clean_text, clean_subjects, year_of)
from utils.paths import (PATH_PERSONAL_DATA, PATH_OLDP_DATA, 
                         HANDCRAFTED_BOOKS_PATH, CACHE_PATH)


##### PARAMETERS ##### ---------------------------------------------------------
DATE = r"(\d{4}-\d{2}-\d{2}|latest)"  # Pattern for date in OLDP dumps.
PATTERNS = {
    "all": re.compile(rf"^ol_dump_{DATE}\.txt\.gz$"),
    "ratings": re.compile(rf"^ol_dump_ratings_{DATE}\.txt\.gz$")
}
FILES_LABELS = {
    "all": "all-types dump", 
    "ratings": "ratings dump", 
}

# Some works are dropped afterwards (no English description...): fetch a few
# more than --top so that the final list can still reach --top books.
LIMIT_HEADROOM = 1.3
DEF_TOP = 15000
DEF_MIN_CHAR = 50
DEF_MAX_CHAR = 999
DEF_MIN_POP = 0
MAX_DAYS = 65


##### Helper ##### -------------------------------------------------------------
def parse_args_and_check():
    """
        Fetches user arguments from terminal

        Returns
        -------
        (args, ol_files): user arguments and paths to OLDP local dumps.
    """
    # Fetch arguments from user command
    ap = argparse.ArgumentParser(
        description=__doc__, 
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--data-dir", 
        default=PATH_OLDP_DATA, type=Path)
    ap.add_argument(
        "--out-dir", 
        default=PATH_PERSONAL_DATA, type=Path)
    ap.add_argument(
        "--top", 
        default=DEF_TOP, type=int, 
        help=f"number of books to keep (default is {DEF_TOP})")
    ap.add_argument(
        "--min-desc", 
        default=DEF_MIN_CHAR, type=int, 
        help=f"min description length in chars (default is {DEF_MIN_CHAR})")
    ap.add_argument(
        "--max-desc", 
        default=DEF_MAX_CHAR, type=int, 
        help=f"truncate summaries to this length (default is {DEF_MAX_CHAR})")
    ap.add_argument(
        "--min-popularity", 
        default=DEF_MIN_POP, type=int, 
        help=f"min ratings + reading-log entries (default is {DEF_MIN_POP})")
    ap.add_argument(
        "--candidates", 
        default=None, type=int,
        help="most-rated works to extract from the dump (default: 4 x --top, min 20000)")
    ap.add_argument(
        "--max-age-days", 
        default=MAX_DAYS, type=int, 
        help=f"warning if dumps are older than {MAX_DAYS} days")
    ap.add_argument(
        "--no-embeddings", 
        action="store_true",
        help="skip the sentence embeddings step (they will be computed when the Recommendation page first opens)")
    args = ap.parse_args()
    if args.candidates is None:
        args.candidates = max(4 * args.top, 20_000)

    # Verify that all-types dump and ratings dump are both available
    ol_files = {
        pattern_key: find_dump(args.data_dir, PATTERNS[pattern_key]) 
        for pattern_key in PATTERNS
    }

    for kind in ("all", "ratings"):
        if not ol_files[kind]:
            sys.exit(
                f"✗ No {FILES_LABELS[kind]} found in '{args.data_dir}'." 
                " Download from 'https://openlibrary.org/developers/dumps'\n")

    # Show a warning if files are outdated
    print("Checking dump freshness (limit: %d days)..." % args.max_age_days)
    for _, file_path in ol_files.items():
        if file_path:
            _ = check_age(file_path, args.max_age_days)

    return args, ol_files

def compute_embeddings(books, out_dir):
    """
        Precomputes the sentence embeddings of the catalogue, so that the
        Recommendation page does not have to (same cache file, same texts,
        same hashes: see utils.embeddings.embed_books).

        Books of handcrafted_books.json (if any) are included: embed_books()
        rewrites its cache with exactly the books it is given, so leaving them
        out would erase their vectors.

        Parameters
        ----------
        books: list[dict]
            The books just written to books.json.
        out_dir: Path
            The folder where the cache (embeddings.npz) is saved.
    """
    catalogue = {b["id"]: b for b in books}
    handcrafted_path = out_dir / HANDCRAFTED_BOOKS_PATH.name
    if handcrafted_path.exists():
        try:
            extra = json.loads(handcrafted_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            extra = []
        for b in extra if isinstance(extra, list) else []:
            if b.get("id") and b.get("title") and len(b.get("summary") or "") >= 50:
                catalogue.setdefault(b["id"], b)

    print(f"Computing sentence embeddings of {len(catalogue):,} books...")
    try:
        with fraction_progress("  embeddings") as update:
            embed_books(catalogue, cache_path=out_dir / CACHE_PATH.name, progress=update)
    except EmbeddingsUnavailable as exc:
        print(f"[WARNING]: embeddings not computed ({exc})\n"
              "[WARNING]: they will be computed when the Recommendation page is first opened.")

def main():
    args, ol_files = parse_args_and_check()

    # Extraction from database
    args.out_dir.mkdir(parents=True, exist_ok=True)
    database = args.out_dir / f"ol_extract_{ol_files['all'].name.replace('.txt.gz', '')}_c{args.candidates}.duckdb"
    con = build_slim(database, ol_files["all"], ol_files["ratings"], args.candidates)

    # Adding popularity
    print("Adding ratings...")
    with progress("  ratings dump", ol_files["ratings"].stat().st_size):
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE rat AS
            SELECT work_key, count(*) AS n, avg(rating) AS avg_rating
            FROM read_csv('{sql_path(ol_files["ratings"])}', delim='\t', header=false, quote='', escape='',
                        columns={{'work_key':'VARCHAR','edition_key':'VARCHAR','rating':'DOUBLE','d':'VARCHAR'}},
                        ignore_errors=true)
            GROUP BY work_key
        """)
    con.execute("CREATE OR REPLACE TEMP TABLE logs (work_key VARCHAR, n BIGINT)")

    # Extraction of english works
    print("Selecting English works with descriptions, ranked by popularity...")
    with progress("  querying"):
        rows = con.execute("""
            WITH eng AS (SELECT DISTINCT work_key FROM slim
                         WHERE type = '/type/edition' AND work_key IS NOT NULL),
            ed_desc AS (SELECT work_key, max_by(description, length(description)) AS description
                        FROM slim WHERE type = '/type/edition' AND description IS NOT NULL
                        GROUP BY work_key)
            SELECT * FROM (
              SELECT w.key, w.title, w.subtitle,
                     w.description AS work_description,
                     d.description AS edition_description,
                     w.first_sentence, w.subjects, w.author_keys, w.first_publish_date, w.covers,
                     COALESCE(r.n, 0) AS n_ratings, r.avg_rating,
                     COALESCE(r.n, 0) + COALESCE(l.n, 0) AS popularity
              FROM slim w
              JOIN eng ON eng.work_key = w.key
              LEFT JOIN ed_desc d ON d.work_key = w.key
              LEFT JOIN rat  r ON r.work_key = w.key
              LEFT JOIN logs l ON l.work_key = w.key
              WHERE w.type = '/type/work' AND w.title IS NOT NULL
            ) WHERE (length(work_description) >= $min OR length(edition_description) >= $min)
                AND popularity >= $minpop
            ORDER BY popularity DESC
            LIMIT $lim""", {"min": args.min_desc, "minpop": args.min_popularity, "lim": int(args.top * LIMIT_HEADROOM)}).fetchall()
    print(f"Found {len(rows)} candidate works!")

    # Cleaning collected rows
    print("Fetching authors and editions...")
    author_keys = sorted({k for row in rows for k in (row[7] or [])})
    authors = {k: (n, pn, alt or []) for k, n, pn, alt in con.execute(
        """
            SELECT key, name, personal_name, alternate_names FROM slim
                WHERE type = '/type/author' AND list_contains(?, key)
        """, 
        [author_keys]).fetchall()}
    print("Cleaning editions...")
    editions = aggregate_editions(con, [r[0] for r in rows], max_per_work=50)
    print(f"{len(editions):,} works aggregated")

    # Creating books json
    books = []
    # First English description of each work: its own, else its English editions'
    print("Checking languages...")
    picked = {}
    for row in track(rows, "  descriptions", unit="works"):
        for src, desc in (("work", row[3]), ("edition", row[4])):
            cleaned = keep_english(clean_text(desc, args.max_desc), args.min_desc)
            if cleaned:
                picked[row[0]] = (cleaned, src)
                break
    # Words of the English summaries: the language detector is unreliable on
    # one-word subjects, so these serve to spot foreign ones ("Novela").
    vocab = english_vocabulary(text for text, _ in picked.values())

    print("Saving results...")
    skipped = {"summary": 0, "title": 0, "duplicate": 0}
    seen = set()
    for (key, title, subtitle, work_desc, ed_desc, fsent, subjects, akeys, fpd, covers,
         n_ratings, avg, pop) in rows:
        summary, desc_src = picked.get(key, ("", None))
        if not summary:                             # junk or not English
            skipped["summary"] += 1
            continue
        ed = editions.get(key)
        # Work titles can be in another language ("O Alquimista"): use the
        # title of the English editions instead.
        title = english_title(title, ed["titles"] if ed else Counter())
        if not title:
            skipped["title"] += 1
            continue
        auth = [{"name": authors[k][0], "personal_name": authors[k][1], "alternate_names": authors[k][2]}
                for k in (akeys or []) if k in authors and authors[k][0]]
        cover_ids = [c for c in (covers or []) if c and c > 0]
        years = ed["years"] if ed else []
        author = ", ".join(a["name"] for a in auth[:3]) or "Unknown"
        dedup_key = (title.strip().lower(), author.strip().lower())
        if dedup_key in seen:           # rows are sorted by popularity: the first one is the best one
            skipped["duplicate"] += 1
            continue
        seen.add(dedup_key)
        books.append({
            "id": key.split("/")[-1],               # e.g. OL45883W (stable across dumps)
            "title": title.strip(),
            "subtitle": subtitle,
            "author": author,
            "authors": auth,
            "subjects": clean_subjects(subjects, limit=40, vocab=vocab),
            "summary": summary,
            "description_source": desc_src,
            "first_sentence": keep_english(clean_text(fsent or (ed["first_sentence"] if ed else ""), 300)) or None,
            "notes": keep_english(clean_text(ed["notes"], 300)) or None if ed and ed["notes"] else None,
            "series": [s for s, _ in ed["series"].most_common(3)] if ed else [],
            "publishers": [p for p, _ in ed["pubs"].most_common(5)] if ed else [],
            "first_publish_date": fpd,
            "first_publish_year": year_of(fpd) or (min(years) if years else None),
            "pages": int(statistics.median(ed["pages"])) if ed and ed["pages"] else None,
            "covers": cover_ids[:3],
            "cover_url": f"https://covers.openlibrary.org/b/id/{cover_ids[0]}-M.jpg" if cover_ids else None,
            "identifiers": dict(ed["ids"]) if ed else {},
            "n_english_editions": ed["n"] if ed else 0,
            "n_ratings": n_ratings,
            "avg_rating": round(avg, 2) if avg is not None else None,
            "popularity": pop,
        })
        if len(books) == args.top:
            break
    print(f"Skipped {skipped['summary']} without an English summary, "
        f"{skipped['title']} without an English title, "
        f"{skipped['duplicate']} duplicates.")
    if len(books) < args.top:
        print(f"[WARNING]: only {len(books)} books kept (--top {args.top}). "
            "Raise LIMIT_HEADROOM or lower --min-desc.")

    # Write results
    books = canonicalize_subjects(books)
    out_file = args.data_dir / "books.json"
    out_file.write_text(
        json.dumps(books, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"\n✓ Wrote {len(books)} books to {out_file} "
        f"({out_file.stat().st_size / 1e6:.1f} MB)")

    # write parameters
    meta = {
        "data_dir": str(args.data_dir),
        "out_dir": str(args.out_dir),
        "top": args.top,
        "min_desc": args.min_desc,
        "max_desc": args.max_desc,
        "min_popularity": args.min_popularity,
        "dump": ol_files["all"].name
    }
    (args.out_dir / "books_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    if not args.no_embeddings:
        compute_embeddings(books, args.data_dir)


##### MAIN ##### ---------------------------------------------------------------
if __name__ == "__main__":
    main()
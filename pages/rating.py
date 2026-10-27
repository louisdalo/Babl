"""
    The annotation page (shown through app.py): streamlit run app.py
    1. Reads books.json & handcrafted_books.json (creates the latter if missing)
    2. Shows a book's features (or the book you searched for)
    3. Registers reviews to labels.csv file
    4. Lets you edit or delete past ratings in the "My ratings" tab
    5. Lets you add manually a book to the dataset using the "Add book" tab
"""


##### IMPORTS ##### ------------------------------------------------------------
import csv
import json
import math
import time
import random
import unicodedata
import pandas as pd
import streamlit as st
from datetime import datetime
from collections import Counter
from uuid import uuid4

from utils.paths import BOOKS_PATH, LABELS_PATH, HANDCRAFTED_BOOKS_PATH
from utils.data import (
    FIELDS, TARGET, HANDCRAFTED_MIN_CHAR, append_label, ensure_handcrafted_file, 
    load_books, read_labels)
from utils.cover import cover_src, cover


##### PARAMETERS ##### ---------------------------------------------------------
QUEUE_SIZE = 3           # books picked ahead of time so their covers can be preloaded
SEARCH_MAX_RESULTS = 10  # books listed at once under the search bar
N_TOP_GENRES = 3

# button key -> (label, read)
ACTIONS = {
    "liked": (1.0, 1),
    "meh": (0.5, 1),
    "disliked": (0.0, 1),
    "interested": (1.0, 0),
    "not_interested": (0.0, 0),
}

# button key -> text shown in the "My ratings" tab
STATUS = {
    "liked": "Read · Liked",
    "meh": "Read · Neutral",
    "disliked": "Read · Disliked",
    "interested": "Not read · Interested",
    "not_interested": "Not read · Not for me",
}
ACTION_BY_STATUS = {text: key for key, text in STATUS.items()}
ACTION_BY_PAIR = {pair: key for key, pair in ACTIONS.items()}  # (label, read) -> key

# sampling strategy -> descriptions for the tooltip
# first strategy is the default strategy
MODE_TOP = "Most popular first"
MODE_RANDOM = "Random pick"
MODE_DIVERSE = "Evolve with annotations"

MODES = {
    MODE_TOP: "Shows the most rated book, then the next one, and so on.",
    MODE_RANDOM: "Shows books picked at random in the available dataset.", # not entirely at random, but the user does not need to know the details
    MODE_DIVERSE: "Shows books from the genres you have rated the least  so far. " +
                  "Best mode to cover your tastes quickly.",
}

##### FUNCTIONS ##### ----------------------------------------------------------
### Cached functions -----------------------------------------------------------
@st.cache_data(show_spinner=False)  # cache for small object
def popular_pool(min_pop: int):
    """
        Orders books by popularity.

        Parameters
        ----------
        max_days: int
            The minimum popularity of a book to add it to the ranking.

        Returns
        ------
        A list of book's IDs.
    """
    return [bid for bid, b in load_books().items() if b["popularity"] >= min_pop]

@st.cache_data(show_spinner=False)  # cache for small object
def suggestions():
    """
        Scans the entire book dataset and returns a list of unique authors, 
        series, and subjects that are used as suggestions in the “Add book” form.
    
        Returns
        -------
        A set of three elements with: unique authors, series and tags.
    """
    authors, series, tags = set(), set(), set()
    for book in load_books().values():
        names = [a["name"] for a in book.get("authors") or [] if isinstance(a, dict) and a.get("name")]
        authors.update(n.strip() for n in names)
        if not names and book.get("author"):
            authors.add(str(book["author"]).strip())
        for item in book.get("series") or []:
            if isinstance(item, str) and item.strip():
                series.add(item.strip())
            elif isinstance(item, dict) and item.get("name"):
                series.add(str(item["name"]).strip())
        tags.update(t.strip() for t in book.get("subjects") or [] if isinstance(t, str) and t.strip())
    return {
        "authors": sorted(authors, key=str.casefold),
        "series": sorted(series, key=str.casefold),
        "tags": sorted(tags, key=str.casefold),
    }


### Labels ---------------------------------------------------------------------
def write_labels(rows):
    """
        Adds a row (aka a new label) into LABELS_PATH file directly.

        Parameters
        ----------
        rows: iterable
            A list of lists containing strings, see csv.writerows 
            (writerows takes 2-dimensional data (multiple rows)).
    """
    tmp = LABELS_PATH.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(LABELS_PATH)

def update_labels(changes, deletions):
    """
        Reads labels, removes deleted entries and add/update other labels 
        directly in file.

        Parameters
        ----------
        changes: dict
            Set of book_ids with their attributed ACTIONS

        deletions: set
            Set of book_ids
    """
    now = int(time.time())
    rows = []
    for r in read_labels():
        if r["book_id"] in deletions:
            continue
        if r["book_id"] in changes:
            label, read = ACTIONS[changes[r["book_id"]]]
            r.update(label=label, read=read, timestamp=now)
        rows.append(r)
    write_labels(rows)

def status_of(row):
    """
        A function that turns into STATUS the ACTIONS of a row.

        Parameters
        ----------
        row: dict
            Set of values (contains "label" and "row")
    """
    action = ACTION_BY_PAIR.get((float(row["label"]), int(row["read"])))
    return STATUS.get(action)

def undo_last():
    """
        Reads last row in LABELS_PATH and removes it.
    """
    rows = read_labels()
    if not rows:
        return None
    last = rows.pop()
    write_labels(rows)
    return last


### Search ---------------------------------------------------------------------
def normalize(text: str):
    """
        Normalizes (agressively) a text into a simplified, consistent format.

        Parameters
        ----------
        text : str
            A text.       

        Returns
        ------
        A (cleaned) string.
    """
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in text if not unicodedata.combining(c)).casefold()

@st.cache_resource  # cache for large object
def search_index():
    """
        Creates a searchable text string for each book.
        The string contain all text from title + subtitle + author.

        Returns
        -------
        A dict
    """
    return {
        bid: normalize(" ".join(filter(None, [b["title"], b.get("subtitle"), b.get("author")])))
        for bid, b in load_books().items()
    }

def search_books(query: str, limit: int=SEARCH_MAX_RESULTS):
    """
        Divides a query into tokens, and find matching books in search_index()

        Parameters
        ----------
        query: str
            A research query of the user.
        limit: int
            A number of results to return

        Returns
        -------
        A list of books
    """
    tokens = normalize(query).split()
    if not tokens:
        return [], 0
    books = load_books()
    hits = [bid for bid, text in search_index().items() if all(t in text for t in tokens)]
    hits.sort(key=lambda bid: books[bid]["popularity"], reverse=True)
    return hits[:limit], len(hits)


### Sampling -------------------------------------------------------------------
def top_genres(book, n: int=N_TOP_GENRES):
    """
        Extracts a book genres and subjects (if one or both exists)

        Parameters
        ----------
        book: dict
            A book dict with fields "genres" and "subjects"
        n: int
            The number of genres or subjects to return

        Returns
        -------
        A list of strings
    """
    return [g.lower() for g in (book.get("genres") or book.get("subjects") or [])[:n]]

def pick_next(mode: str, min_pop: int, rated_ids, skipped, exclude=frozenset()):
    """
        Chooses the next book based on given mode.

        Parameters
        ----------
        mode: str
            The name of a mode.
        min_pop: int
            The minimum popularity. Selects only books with popularity >= min_pop.
        rated_ids: list
            A list of book IDs (those already rated by the user).
        skipped: list
            A list of book IDs (those already skipped by the user).
        exclude: list.
            A list of book IDs (usually, to exclude the book being shown/in queue).

        Returns
        -------
        A list of strings
    """
    if mode not in MODES:
        raise ValueError(f"Unknown sampling mode: {mode!r}")
    books = load_books()
    pool = [i for i in popular_pool(min_pop) if i not in rated_ids and i not in skipped and i not in exclude]
    if not pool:
        return None
    if mode == MODE_TOP:
        return max(pool, key=lambda i: books[i]["popularity"])
    weights = [math.sqrt(max(books[i]["popularity"], 1)) for i in pool]
    cands = list({random.choices(pool, weights)[0] for _ in range(40)})
    if mode == MODE_RANDOM:
        return random.choice(cands)
    seen = Counter()
    for bid in rated_ids:
        if bid in books:
            seen.update(top_genres(books[bid]))
    def coverage(i):
        g = top_genres(books[i])
        return sum(seen[x] for x in g) / max(len(g), 1)
    cands.sort(key=coverage)
    return random.choice(cands[:5])


##### Streamlit ##### ----------------------------------------------------------
### Initial states -------------------------------------------------------------
ss = st.session_state           # This values are saved for the whole session
ss.setdefault("current", None)           # No book displayed
ss.setdefault("skipped", set())          # No book skipped
ss.setdefault("flash", "")               # No message in flash memory
ss.setdefault("queue", [])               # No books in queue
ss.setdefault("sig", None)               # Stores mode and min_pop chosen by the user
ss.setdefault("search", "")              # Search input (empty)
ss.setdefault("editor_version", 0)       # Bumped after each change in editor
ss.setdefault("add_version", 0)          # Bumped after each added book
ss.setdefault("activate_settings", True) # Settings activation

st.title("Annotations")
tab_rate, tab_edit, tab_add = st.tabs(
    [
        ":material/menu_book: Rate books",
        ":material/edit_note: Books you rated",
        ":material/library_add: Add book to database"
    ],
    on_change="rerun"
)

ss.activate_settings = not tab_rate.open

# Load JSONs
ensure_handcrafted_file() 
if not BOOKS_PATH.exists():
    st.error(f"Can't find {BOOKS_PATH.resolve()}. Run the app from the folder containing books.json.")
    st.stop()
books = load_books()
labels = read_labels()
rated_ids = {r["book_id"] for r in labels}
reviews_by_id = {r["book_id"]: r for r in labels}

# Create app Sidebar
with st.sidebar:
    st.header("Settings")

    # Make modes radio buttons
    mode = st.radio(
        "Next book to show", list(MODES),
        help="\n\n".join(f"**{name}**: {desc}" for name, desc in MODES.items()),
        disabled = ss.activate_settings
    )

    # Make popularity slider
    max_pop = max((b["popularity"] for b in books.values()), default=0)
    min_pop = st.slider(
        "Minimal number of reviews", 0, min(max_pop, 5000), 1, step=1,
        help="The interface shows books that have a recorded number "
             "of reviews equal or superior to the one selected on the slider.",
        disabled = ss.activate_settings)

    # Show global label infos
    st.caption(f"{len(popular_pool(min_pop)):,} books in pool / {len(books):,} total")
    st.header("Progress")
    n = len(labels)
    st.progress(min(n / TARGET, 1.0), text=f"{n} / {TARGET} labels")
    pos = sum(float(r["label"]) == 1.0 for r in labels)
    neg = sum(float(r["label"]) == 0.0 for r in labels)
    neu = n - pos - neg
    st.write(f":material/thumb_up: {pos}   :material/sentiment_neutral: {neu}   :material/thumb_down: {neg}")
    st.write(f"Read: {sum(int(r['read']) for r in labels)}  ·  Not read: {n - sum(int(r['read']) for r in labels)}")
    if n >= TARGET:
        st.success("Enough labels: see the Recommendation page", icon=":material/celebration:")


### Actions --------------------------------------------------------------------
def fill_queue():
    """
        Fills the queue with books that were not already rated/skipped/excluded.

        Updates
        -------
        ss.queue
    """
    rated = {r["book_id"] for r in read_labels()}
    while len(ss.queue) < QUEUE_SIZE:
        nxt = pick_next(mode, min_pop, rated, ss.skipped, exclude=set(ss.queue) | {ss.current})
        if nxt is None:
            break
        ss.queue.append(nxt)

def next_book():
    """
        Adds a book to the queue, switch to next book in queue and refills queue.

        Updates 
        -------
        ss.current 
        ss.queue
    """
    fill_queue()
    ss.current = ss.queue.pop(0) if ss.queue else None
    fill_queue()

def rate(action: str):
    """
        When a book is rated, adds label to file and auto-switches to next book.

        Parameters
        ----------
        action: str
            One of the actions in ACTIONS.

        Updates 
        -------
        ss.current
        ss.queue
        LABELS_PATH.
    """
    label, read = ACTIONS[action]
    append_label(books[ss.current], label, read)
    next_book()

def skip():
    """
        Skips current book.

        Updates 
        -------
        ss.skipped
        ss.current
        ss.queue
    """
    ss.skipped.add(ss.current)
    next_book()

def undo():
    """
        Undo last book added to labels and resets displayed book to this one.

        Updates 
        -------
        ss.current
        ss.queue
        ss.flash
        LABELS_PATH
    """
    last = undo_last()
    if last:
        ss.current = last["book_id"]
        ss.flash = f"Undid: {last['title']}"

def pick(book_id: str):
    """
        Replaces current book displayed by the one given.

        Parameters
        ----------
        book_id:
            Do I really need to explain?
        
        Updates 
        -------
        ss.current
        ss.queue
        ss.skipped
        ss.search
    """
    if ss.current and ss.current != book_id:
        ss.queue.insert(0, ss.current)
    if book_id in ss.queue:
        ss.queue.remove(book_id)
    ss.skipped.discard(book_id)
    ss.current = book_id
    ss.search = ""


# Initialisation securities
sig = (mode, min_pop)
if ss.sig != sig:
    if ss.sig is not None:
        ss.queue = []
        ss.current = None
    ss.sig = sig
if ss.current is None or ss.current not in books:
    next_book()


### User Interface -------------------------------------------------------------
def render_search():
    """
        Renders a search section
    """
    st.text_input(
        "Search", key="search", icon=":material/search:", label_visibility="collapsed",
        placeholder="Search a book to rate by title or author (e.g. \"tolkien\", \"dune\")",
    )
    query = ss.search.strip()

    # no research under 3 char.
    if len(query) < 2:                  
        return

    # if more, do the research.
    hits, total = search_books(query)   
    if not hits:
        st.caption("No book found.")
        return

    # Row styling: injected once, shared by all result rows
    st.markdown(
        """
        <style>
        [class*="st-key-result_"] {
            padding: 12px;
            border-radius: 12px;
            border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        }
        [class*="st-key-result_"]:hover {
            background-color: rgba(128, 128, 128, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # If one or more books found, display them.
    with st.container(border=True):
        for i, bid in enumerate(hits):
            r = books[bid]

            with st.container(key=f"result_{bid}", border=False):
                info, action = st.columns([5, 1], vertical_alignment="center")
                meta = [r.get("author") or "Unknown author"]
                if r.get("first_publish_year"):
                    meta.append(str(r["first_publish_year"]))
                info.markdown(f"**{r['title']}**  \n{' · '.join(meta)}")
                if bid in rated_ids:
                    info.caption(f"Already rated: {status_of(reviews_by_id[bid])} (edit it in the \"Books you rated\" tab)")
                    action.button("Rated", key=f"pick_{bid}", disabled=True, width="stretch")
                elif bid == ss.current:
                    action.button("Showing", key=f"pick_{bid}", disabled=True, width="stretch")
                else:
                    action.button("Rate", key=f"pick_{bid}", icon=":material/arrow_forward:",
                                on_click=pick, args=(bid,), width="stretch")
                    
    if total > len(hits):
        st.caption(f"{total - len(hits):,} more matches: type more of the title or the author to narrow it down.")

def render_card():
    """
        Renders a book info card and rating options.
    """
    # No current book, display a message
    if ss.current is None:
        st.info("No more books in this pool. Lower the popularity threshold in the sidebar, or search for a book.")
        return

    # If there is a book to display, show all its infos.
    b = books[ss.current]
    with st.container(height=450, border=True):
        left, right = st.columns([1, 3], vertical_alignment="center")
        with left:
            cover(b.get("cover_url"), b["title"])
        with right:
            st.subheader(b["title"] + (f": {b['subtitle']}" if b.get("subtitle") else ""))
            meta = [b.get("author") or "Unknown author"]
            if b.get("first_publish_year"):
                meta.append(str(b["first_publish_year"]))
            if b.get("pages"):
                meta.append(f"{b['pages']} pages")
            st.write(" · ".join(meta))
            if b.get("avg_rating"):
                st.caption(f":material/star: {b['avg_rating']} avg · {b['n_ratings']:,} reviews on Open Library")
            genres = [g for g in (b.get("genres") or b.get("subjects") or []) if g.isascii()][:8]
            if genres:
                st.write(" ".join(f"`{g}`" for g in genres))
            st.write(b["summary"])
            if b.get("first_sentence"):
                st.markdown(f"*“{b['first_sentence']}”*")

    # Under the card for the book, show rating options
    col1, col2 = st.columns(2, border=True)

    with col1:
        st.subheader("I've read it", text_alignment="center", anchor=False)
        st.button("Liked", icon=":material/thumb_up:", on_click=rate, args=("liked",), width="stretch")
        st.button("No opinion", icon=":material/sentiment_neutral:", on_click=rate, args=("meh",), width="stretch")
        st.button("Disliked", icon=":material/thumb_down:", on_click=rate, args=("disliked",), width="stretch")

    with col2:
        st.subheader("I haven't read it", text_alignment="center", anchor=False)
        st.space("small")
        st.button("Interested", icon=":material/favorite:", on_click=rate, args=("interested",), width="stretch")
        st.button("Not for me", icon=":material/block:", on_click=rate, args=("not_interested",), width="stretch")

    _, c6, c7, _ = st.columns([1, 2, 2, 1])
    c6.button("Skip / Unknown)", icon=":material/skip_next:", on_click=skip, width="stretch")
    c7.button("Undo last annotation", icon=":material/undo:", on_click=undo, width="stretch")

def render_history():
    """
        Shows labels in a table (editable).
    """
    # nothing in labels, nothing to show
    if not labels:
        st.info("You haven't rated any book yet.")
        return

    # If one or more labels, make table
    f1, f2 = st.columns([3, 2])
    query = f1.text_input(
        "Filter", key="history_query", icon=":material/filter_alt:", label_visibility="collapsed",
        placeholder="Filter by title or author",
    )
    show = f2.selectbox("Show", ["All", *STATUS.values()], label_visibility="collapsed")
    table = pd.DataFrame([
        {
            "book_id": r["book_id"],
            "Title": r["title"],
            "Author": books.get(r["book_id"], {}).get("author") or "",
            "Rating": status_of(r),
            "Rated on": datetime.fromtimestamp(int(r["timestamp"])),
            "Delete": False,
        }
        for r in labels
    ])
    table = table.sort_values("Rated on", ascending=False, kind="stable")
    table = table.drop_duplicates("book_id").set_index("book_id")
    tokens = normalize(query).split()
    haystack = (table["Title"] + " " + table["Author"]).map(normalize)
    mask = haystack.map(lambda text: all(t in text for t in tokens))
    if show != "All":
        mask &= table["Rating"] == show
    view = table[mask]
    if view.empty:
        st.caption("No rating matches this filter.")
        return
    edited = st.data_editor(
        view,
        key=f"history_{ss.editor_version}_{normalize(query)}_{show}",
        hide_index=True,
        width="stretch",
        height=min(80 + 35 * len(view), 520),
        disabled=["Title", "Author", "Rated on"],
        column_config={
            "Rating": st.column_config.SelectboxColumn("Rating", options=list(STATUS.values()), required=True),
            "Rated on": st.column_config.DatetimeColumn("Rated on", format="YYYY-MM-DD HH:mm"),
            "Delete": st.column_config.CheckboxColumn(
                "Delete", help="Remove this rating: the book can be proposed again."),
        },
    )

    # What happens if the user clicks on "delete"
    deletions = set(edited.index[edited["Delete"]])
    changes = {
        bid: ACTION_BY_STATUS[new]
        for bid, new in edited["Rating"].items()
        if pd.notna(new) and new != view.at[bid, "Rating"] and bid not in deletions
    }
    n_edits = len(changes) + len(deletions)
    st.caption(f"{len(view):,} / {len(table):,} ratings shown. Edits are only written when you press Save.")
    if st.button(f"Save {n_edits} change{'s' if n_edits != 1 else ''}", icon=":material/save:",
                 type="primary", disabled=n_edits == 0):
        update_labels(changes, deletions)
        ss.editor_version += 1
        ss.flash = f"Saved: {len(changes)} edited, {len(deletions)} deleted"
        st.rerun()

@st.fragment
def render_add_book():
    """
        A form to add a book to a personal JSON
    """
    # Create a custom ID for next book
    v = ss.add_version
    def k(name):
        return f"add_{name}_{v}"

    # Load suggestions (for authors, titles and subjects fields)
    sugg = suggestions()

    # Display instructions
    st.caption("Pick an existing value or type a new one, then press Enter to validate it.")

    # Display form 
    with st.form(f"add_book_form_{v}", border=True):
        st.markdown("**Basic information**")
        title = st.text_input("Title *", key=k("title"))
        subtitle = st.text_input("Subtitle", key=k("subtitle"))
        author = st.selectbox("Author", options=sugg["authors"], index=None, accept_new_options=True,
                              placeholder="Pick an existing author or type a new one", key=k("author"))
        summary = st.text_area(f"Summary * (at least {HANDCRAFTED_MIN_CHAR} characters)", height=120, key=k("summary"))
        description_source = st.text_input("Description source", key=k("description_source"))
        first_sentence = st.text_area("First sentence", height=70, key=k("first_sentence"))
        notes = st.text_area("Notes", height=70, key=k("notes"))
        cover_url = st.text_input("Cover URL", key=k("cover_url"))

        st.markdown("**Subjects / tags**")
        tags = st.multiselect("Subjects", options=sugg["tags"], accept_new_options=True,
                              placeholder="Pick existing subjects or type new ones", key=k("tags"))

        st.markdown("**Publication details**")
        c1, c2, c3 = st.columns(3)
        first_publish_date = c1.text_input("First publish date", key=k("first_publish_date"))
        first_publish_year = c2.number_input("First publish year", min_value=0, max_value=3000, value=0,
                                             help="0 means unknown", key=k("first_publish_year"))
        pages = c3.number_input("Pages", min_value=0, max_value=100000, value=0,
                                help="0 means unknown", key=k("pages"))
        publishers_text = st.text_area("Publishers (one per line)", height=70, key=k("publishers"))
        series = st.multiselect("Series", options=sugg["series"], accept_new_options=True,
                                placeholder="Pick existing series or type new ones", key=k("series"))
        covers_text = st.text_input("Cover IDs (comma-separated numbers)", key=k("covers"))

        st.markdown("**Identifiers and statistics**")
        identifiers_json = st.text_area("Identifiers (JSON object)", value="{}", height=80, key=k("identifiers"))
        c1, c2, c3 = st.columns(3)
        n_english_editions = c1.number_input("English editions", min_value=0, value=0, key=k("n_editions"))
        n_ratings = c2.number_input("Number of reviews", min_value=0, value=0, key=k("n_ratings"))
        avg_rating = c3.number_input("Average rating", min_value=0.0, max_value=5.0, value=0.0, step=0.01,
                                     key=k("avg_rating"))
        popularity = st.number_input("Popularity", min_value=0, value=0, key=k("popularity"))
        submitted = st.form_submit_button("Add book to database", type="primary", icon=":material/library_add:")

    # If page left before submission, forget form
    if not submitted:
        return

    # A list of fields that must be completed.
    if not (title or "").strip():
        st.error("Title is required.")
        return
    summary_len = len((summary or "").strip())
    if summary_len < HANDCRAFTED_MIN_CHAR:
        st.error(f"Please enter at least {HANDCRAFTED_MIN_CHAR} characters for the summary ({HANDCRAFTED_MIN_CHAR - summary_len} more needed).")
        return
    try:
        identifiers = json.loads(identifiers_json or "{}")
        if not isinstance(identifiers, dict):
            raise ValueError("Identifiers must be a JSON object.")
        covers = [int(x.strip()) for x in covers_text.split(",") if x.strip()]
    except (json.JSONDecodeError, ValueError) as exc:
        st.error(f"Check the identifiers JSON or cover IDs: {exc}")
        return

    # Format values fetched from the form and add to file.
    author_value = (author or "").strip() or None
    authors = [{"name": author_value, "personal_name": None, "alternate_names": []}] if author_value else []
    book = {
        "id": "HC-" + uuid4().hex[:12],
        "title": title.strip(), "subtitle": (subtitle or "").strip() or None,
        "author": author_value, "authors": authors, "subjects": list(tags),
        "summary": summary.strip(), "description_source": (description_source or "").strip() or None,
        "first_sentence": (first_sentence or "").strip() or None, "notes": (notes or "").strip() or None,
        "series": list(series), "publishers": [x.strip() for x in publishers_text.splitlines() if x.strip()],
        "first_publish_date": (first_publish_date or "").strip() or None,
        "first_publish_year": int(first_publish_year) or None, "pages": int(pages) or None,
        "covers": covers, "cover_url": (cover_url or "").strip() or None,
        "identifiers": identifiers, "n_english_editions": int(n_english_editions),
        "n_ratings": int(n_ratings), "avg_rating": float(avg_rating) or None,
        "popularity": int(popularity),
    }
    ensure_handcrafted_file()
    with open(HANDCRAFTED_BOOKS_PATH, encoding="utf-8") as f:
        handcrafted = json.load(f)
    if not isinstance(handcrafted, list):
        st.error("handcrafted_books.json must contain a JSON list. Please fix the file before adding books.")
        return
    handcrafted.append(book)
    tmp = HANDCRAFTED_BOOKS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(handcrafted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(HANDCRAFTED_BOOKS_PATH)
    ss.add_version += 1             # fresh, empty form
    ss.flash = f"Added ‘{book['title']}’ to handcrafted_books.json."
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()                      # full rerun: the new book must appear everywhere


# Style of buttons (no text wrapping mainly)
st.markdown(
    """<style>
    div[data-testid="stButton"] button { height: 3.2rem; white-space: nowrap; }
    </style>""",
    unsafe_allow_html=True,
)
if ss.flash:
    st.toast(ss.flash)
    ss.flash = ""

# Organise tabs
with tab_rate:
    st.text(f"Annotate books through this interface. Once you get to {TARGET}+ "
            ", you can move to the Recommendation or Masse Critique pages.")
    render_search()
    render_card()
with tab_edit:
    st.text(f"This tab shows previous annotations. You can delete labels "
            "if necessary.")
    render_history()
with tab_add:
    st.markdown(f"This app uses [`Open Library`](https://openlibrary.org/) for "
                "its dataset. It's a participative dataset (which is nice): "
                "users can add and modify records. Unfortunately, it also means "
                "that the database can contain mistakes, or miss books "
                "(which is less nice). To counter this, if you know a book "
                "that you really want to rate but can't find in the database "
                "(see search bar in the `Rate books` tab), then you can add it "
                "manually here.\n\nThe more infos you give, the more useful is "
                "your addition!")
    render_add_book()

# Placeholder for book image
preload = "".join(
    f'<img src="{cover_src(books[i]["cover_url"])}" style="display:none" width="1" height="1">'
    for i in ss.queue
    if i in books and cover_src(books[i].get("cover_url"))
)
if preload:
    st.markdown(preload, unsafe_allow_html=True)
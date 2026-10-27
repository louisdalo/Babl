"""A set of functions to handle books.json and labels.csv files."""

##### IMPORTS ##### ------------------------------------------------------------
import csv
import json
import time
import streamlit as st
from pathlib import Path
from uuid import uuid4

from utils.subjects import canonicalize_subjects
from utils.paths import (BOOKS_PATH, HANDCRAFTED_BOOKS_PATH, 
                         LABELS_PATH, BOOKS_META_PATH)


##### PARAMETERS ##### ---------------------------------------------------------
FIELDS = ["book_id", "label", "read", "timestamp", "title"] # Fields for books in LABELS_PATH
TARGET = 50     # labels needed before recommendations are available
HANDCRAFTED_MIN_CHAR = 50


##### FUNCTIONS ##### ----------------------------------------------------------
def _read_list(path):
    """Reads a json from a Path and returns it if it is a list."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []

def _dedup_key(b):
    """
        Formats a title and authors from a book: removes spaces and transforms 
        to lowercase

        Parameters
        ----------
        b: dict
            Contains the fields "title" and "authors".
        
        Returns
        -------
        A tuple (title, author)
    """
    return (b["title"].strip().lower(), (b.get("author") or "").strip().lower())

def ensure_handcrafted_file():
    """Creates HANDCRAFTED_BOOKS_PATH file if it doesn't already exist."""
    HANDCRAFTED_BOOKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not HANDCRAFTED_BOOKS_PATH.exists():
        HANDCRAFTED_BOOKS_PATH.write_text("[]\n", encoding="utf-8")

def file_signature(path: Path):
    """
        Fetches the signature of a file: its creation timestamp and byte size.

        Parameters
        ----------
        path: Path
            The path to a file.

        Returns
        ------
        A tuple (stat.st_mtime_ns, stat.st_size), creation timestamp and byte size.
    """
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_mtime_ns, stat.st_size)

def _min_desc():
    try:
        return json.loads(BOOKS_META_PATH.read_text(encoding="utf-8"))["min_desc"]
    except (OSError, KeyError, json.JSONDecodeError):
        return 50   # fallback if no sidecar (older books.json)

@st.cache_resource  # big object
def load_books():
    """
        Loads and formats books.json + handcrafted_books.json.

        Parameters
        ----------
        min_char: int
            The path to a file.

        Returns
        ------
        A set of books.
    """
    ensure_handcrafted_file()
    books = {}
    for b in _read_list(BOOKS_PATH):                 # already filtered by prepare_data.py
        b["popularity"] = b.get("popularity") or b.get("n_ratings") or 0
        books[b["id"]] = b

    seen = {_dedup_key(b) for b in books.values()}
    for b in _read_list(HANDCRAFTED_BOOKS_PATH):     # validated here
        if not b.get("title") or len(b.get("summary") or "") < HANDCRAFTED_MIN_CHAR:
            continue
        b["popularity"] = b.get("popularity") or b.get("n_ratings") or 0
        b.setdefault("id", "HC-" + uuid4().hex[:12])
        key = _dedup_key(b)
        if key in seen:
            continue
        seen.add(key)
        books[b["id"]] = b

    canonicalize_subjects(books.values())
    return books

def read_labels():
    """
        Fetches labeled data.

        Returns
        -------
        A list of labels and their values
    """
    if not LABELS_PATH.exists():
        return []
    with open(LABELS_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def append_label(book, label, read):
    """
        Appends one rating to labels.csv (creates the file if needed).

        Parameters
        ----------
        book: iterable[dict]
            A book's features (minimal fields: "id" and "title").
        label: int
            If the book was liked or not (rating between 0 and 1).
        read: int
            1 if the book was read, 0 if it wasn't.

        Returns
        ------
        Adds a book to LABELS_PATH file.
    """
    new = not LABELS_PATH.exists()
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "book_id": book["id"], "label": label, "read": read,
            "timestamp": int(time.time()), "title": book["title"],
        })
""" A set of functions to extract and format "subject" fields from books"""

##### IMPORTS ##### ------------------------------------------------------------
import re
import unicodedata
from collections import Counter, defaultdict


##### FUNCTIONS ##### ----------------------------------------------------------
def subject_key(subject: str):
    """
        Formats subjects to be insensitive to case, accents, punctuation, 
        hyphens, "&", "#"...

        e.g.
            "Fantasy", "FANTASY.", " # Fantasy"     -> "fantasy"
            "Self-help" / "Self help"               -> "self help"

        Parameters
        ----------
        subject: string
            The text to format.

        Returns
        ------
        A string
    """
    s = unicodedata.normalize("NFKD", subject or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    s = re.sub(r"[\W_]+", " ", s)           # punctuation, symbols, "_" -> space
    return re.sub(r"\s+", " ", s).strip()

def _display_form(subject: str):
    """
        Removes surrounding junk of string ("& Fantasy" -> "Fantasy"), leaving 
        the inner text untouched.

        Parameters
        ----------
        subject: string
            The text to format.

        Returns
        ------
        A string
    """
    s = re.sub(r"\s+", " ", subject or "").strip()
    return s.strip(" \t&#*-–—:;,./|")

def canonicalize_subjects(books):
    """
        Merges the subjects that only differ by formatting, across all books.
        For each group, the most frequent written form is kept (ties: the one
        that starts with a capital letter). Duplicates inside one book are
        removed. Books are modified in place.

        Parameters
        ----------
        books: iterable[dict]
            Books with a "subjects" list.

        Returns
        -------
        The same books (as a list).
    """
    books = list(books)
    forms = defaultdict(Counter)
    for b in books:
        for s in b.get("subjects") or []:
            key = subject_key(s)
            disp = _display_form(s)
            if key and disp:
                forms[key][disp] += 1

    canonical = {
        key: max(counter.items(),
                 key=lambda kv: (kv[1], kv[0][:1].isupper() and not kv[0].isupper(), kv[0]))[0]
        for key, counter in forms.items()
    }

    for b in books:
        seen, merged = set(), []
        for s in b.get("subjects") or []:
            key = subject_key(s)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(canonical[key])
        b["subjects"] = merged
    return books
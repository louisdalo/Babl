"""A set of function to standardise series and author's names."""

##### IMPORTS ##### ------------------------------------------------------------
import re
import unicodedata
from collections import Counter, defaultdict


##### PARAMETERS ##### ---------------------------------------------------------
# Generic names often use in series names
_NUMBER_WORDS = "one|two|three|four|five|six|seven|eight|nine|ten"
_GENERIC = {"part", "book", "volume", "series", "collection", "trilogy", "saga"}
DOMINANT_SHARE = 0.6   # a series needs one author on >= 60% of its books


##### FUNCTIONS ##### ----------------------------------------------------------
def series_key(raw):
    """
        Standardises a book series name into a consistent key by removing
        volume numbers, formatting, accents, and leading articles.

        Examples:
            - "Harry Potter #1", "Harry Potter, #3"            
                -> "harry potter"
          "A song of ice and fire -- bk. 1", "Song of Ice and Fire (2)" 
                -> "hunger games"

        Parameters
        ----------
        raw: str or None
            The original series name to standardise. 

        Returns
        -------
        The standardised string name or "" when nothing meaningful is left.
    """
    s = unicodedata.normalize("NFKD", raw or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    s = re.split(r";| -- | - ", s)[0]                                   # "x ; bk. 3", "x -- 4"
    s = re.sub(r"^\W*(?:book|volume|vol\.?)\s+\w+\s+of\s+", "", s)      # "Book One of X"
    s = re.sub(r"#\s*\d+\w*", " ", s)                                   # "#1"
    s = re.sub(r"\([^)]*\d[^)]*\)", " ", s)                             # "(1)", "(Volume 1)"
    s = re.sub(rf"\b(?:book|bk|volume|vol|part|pt|tome)\b\.?\s*(?:\d+|[ivxl]+|{_NUMBER_WORDS})\b", " ", s)
    s = re.sub(r"[,\s]+\d+\s*$", " ", s)                                # "Twisted, 4"
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(?:the|a|an)\s+", "", s)
    return s if len(s) >= 3 and s not in _GENERIC else ""

def _author_key(name):
    """
        Standardises an author's name into a consistent key by removing spaces, 
        punctuation, formattting...

        Example: 
        'J. K. Rowling', 'J.K. Rowling', 'Rowling, J.K.', 'Joanne Rowling' 
            -> 'rowling j'

        Parameters
        ----------
        name: str or None
            The original author's name to standardise. 

        Returns
        -------
        The standardised string name or "" when nothing meaningful is left.
    """
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    if "," in s:
        last, _, first = s.partition(",")
        s = f"{first} {last}"
    words = re.findall(r"\w+", s)
    if not words:
        return ""
    return f"{words[-1]} {words[0][0]}" if len(words) > 1 else words[0]

def _author_keys(book, fallback):
    """
        Standardises an author's name into a consistent key by removing spaces, 
        punctuation, formattting...

        Example: 
        'J. K. Rowling', 'J.K. Rowling', 'Rowling, J.K.', 'Joanne Rowling' 
            -> 'rowling j'

        Parameters
        ----------
        book: dict
            A book's infos.
        fallback: str
            A default key to fall back to if book as no author.

        Returns
        -------
        Standardised authors of a book (primary author first).
    """
    names = [a.get("name") for a in book.get("authors") or [] if isinstance(a, dict)]
    if not any(names) and book.get("author"):
        names = [n.strip() for n in book["author"].split(",")]
    keys = [k for k in (_author_key(n) for n in names) if k]
    return keys or [fallback]

def series_groups(books):
    """
        Groups books based on their series and authors.

        Normalizes series names and groups books that belong to the same
        recognized series. Books without a qualifying series remain in
        individual groups.

        Publisher collections and shared universes are excluded when
        their normalized series name is associated with multiple authors.
        Series connected through a common book are merged into one group.

        Parameters
        ----------
        book : dict
            Metadatas of a book (extracted from the books dict see 
            utils.data.load_books()).

        Returns
        -------
        A list of integer (group IDs) aligned with list(books), preserving
        the dictionary's iteration order.
    """
    ids = list(books)
    keys, who, counts = {}, {}, defaultdict(Counter)
    for i in ids:
        ks = {series_key(s) for s in books[i].get("series") or [] if isinstance(s, str)}
        ks.discard("")
        keys[i] = ks
        who[i] = _author_keys(books[i], fallback=f"?{i}")
        for k in ks:
            counts[k][who[i][0]] += 1          # vote with the primary author

    # the author each series belongs to (none for publisher collections)
    owner = {}
    for k, c in counts.items():
        a, n = c.most_common(1)[0]
        if n / sum(c.values()) >= DOMINANT_SHARE and not a.startswith("?"):
            owner[k] = a

    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first_of_key = {}
    for i in ids:
        for k in keys[i]:
            if k not in owner or owner[k] not in who[i]:   # any author may match
                continue
            if k in first_of_key:
                parent[find(i)] = find(first_of_key[k])
            else:
                first_of_key[k] = i

    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in ids]
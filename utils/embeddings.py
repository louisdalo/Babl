"""A set of function to turn texts into embeddings """

##### IMPORTS ##### ------------------------------------------------------------
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

from utils.paths import CACHE_PATH

##### PARAMETERS ##### ---------------------------------------------------------
# Model parameters
MODEL_NAME = "all-MiniLM-L6-v2"  # small model (384 dimensions) runs fine on CPU
MAX_CHARS = 1500 # the model only reads the first ~256 tokens anyway

class EmbeddingsUnavailable(RuntimeError):
    """ A custom error raised when a package or model is missing."""


##### FUNCTIONS ##### ----------------------------------------------------------
def book_text(book):
    """
        Turns text in a book's field into a single string.

        Parameters
        ----------
        book: dict
            A set with at least 3 keys: "title", "subtitle" and "summary" 
            containing strings (or "", or None).
        
        Returns
        -------
        a string
    """
    parts = [book.get("title"), book.get("subtitle"), book.get("summary")]
    return ". ".join(p.strip() for p in parts if p)[:MAX_CHARS]

def _digest(text):
    """
        Generates a fixed-length hexadecimal hash from MODEL_NAME combined with 
        the input text. Used to create a compact signature for identifying a 
        or detecting changes.

        Parameters
        ----------
        text: str
            Well, what is there to add?

        Returns
        -------
        hashlib object
    """    
    return hashlib.md5((MODEL_NAME + text).encode("utf-8")).hexdigest()

def _encode_in_subprocess(texts, report):
    """
        Generates text embeddings without importing heavy machine-learning 
        libraries into the main Streamlit process. Uses utils.embeddings_worker 
        so that torch and transformers are never imported in the main Streamlit 
        process. Progress is read from the worker's output.

        Parameters
        ----------
        texts: a set
            Contains a set of texts
        report: a function
            A function that shows a progress bar (see progress in utils.progress)
        
        Returns
        -------
        Embeddings (as numpy array)
    """
    root = Path(__file__).resolve().parent.parent       # project folder: the worker imports `utils`
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, err = Path(tmp) / "texts.json", Path(tmp) / "vectors.npy", Path(tmp) / "stderr.txt"
        src.write_text(json.dumps(texts), encoding="utf-8")
        with open(err, "w", encoding="utf-8") as err_file:
            proc = subprocess.Popen(
                [sys.executable, "-m", "utils.embeddings_worker", str(src), str(dst)],
                cwd=root, stdout=subprocess.PIPE, stderr=err_file, text=True)
            for line in proc.stdout:
                parts = line.split()
                if parts[:1] == ["LOADING"]:
                    report(0.05, "Loading the language model (the first run downloads ~90 MB)...")
                elif parts[:1] == ["PROGRESS"] and len(parts) == 3:
                    done, total = int(parts[1]), int(parts[2])
                    report(0.10 + 0.85 * done / total, f"Computing sentence embeddings: {done:,} / {total:,} books")
            proc.wait()
        if proc.returncode != 0 or not dst.exists():
            details = err.read_text(encoding="utf-8", errors="replace").strip()
            if "sentence_transformers" in details and "No module named" in details:
                raise EmbeddingsUnavailable("sentence-transformers is not installed (pip install sentence-transformers).")
            raise EmbeddingsUnavailable(f"the embedding process failed: ...{details[-400:]}")
        return np.load(dst)

def embed_books(books, encoder=None, cache_path: Path=CACHE_PATH, progress=None, chunk: int=256):
    """
        Creates a unit-length embedding of every book (dict id -> book), as a 
        float32 matrix whose rows follow list(books).

        Embeddings are cached on disk: only the books that are new (or whose
        text changed) are encoded, so the slow step happens once.
        `encoder` (texts -> array) can be injected, e.g. for tests.
        `progress(fraction, message)` is called along the way (optional).

        Parameters
        ----------
        books: dict 
            A set of books and features (see utils.data.load_books() for details)
        encoder: ???
            Optional encoder or embedding model instance.
        cache_path: str
            A path to save embeddings.
        progress: a function
            Optional custom progress bar function.
        chunk: int
            Number of texts encoded at once.

        Returns
        -------
        a numpy float32 matrix
    """
    report = progress or (lambda fraction, message: None)
    report(0.0, "Reading cached embeddings...")
    cache_path = Path(cache_path)
    ids = list(books)
    texts = [book_text(books[i]) for i in ids]
    hashes = [_digest(t) for t in texts]

    cached = {}
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as z:
                if str(z["model"]) == MODEL_NAME:
                    cached = {str(i): (str(h), v) for i, h, v in zip(z["ids"], z["hashes"], z["vectors"])}
        except Exception:           # unreadable cache: recompute
            cached = {}

    todo = [k for k, (i, h) in enumerate(zip(ids, hashes)) if i not in cached or cached[i][0] != h]
    if todo:
        todo_texts = [texts[k] for k in todo]
        if encoder is None:
            vectors = _encode_in_subprocess(todo_texts, report)
        else:                                   # injected encoder (tests): same process
            batches = []
            for start in range(0, len(todo_texts), chunk):
                report(0.10 + 0.85 * start / len(todo_texts),
                       f"Computing sentence embeddings: {start:,} / {len(todo_texts):,} books")
                batches.append(np.asarray(encoder(todo_texts[start:start + chunk]), dtype=np.float32))
            vectors = np.vstack(batches)
        vectors = np.asarray(vectors, dtype=np.float32)
        report(0.96, "Saving embeddings...")
        for k, v in zip(todo, vectors):
            cached[ids[k]] = (hashes[k], v)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path, model=np.array(MODEL_NAME), ids=np.array(ids),
            hashes=np.array(hashes), vectors=np.vstack([cached[i][1] for i in ids]))
    return np.vstack([cached[i][1] for i in ids]).astype(np.float32)
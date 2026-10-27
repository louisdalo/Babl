"""
    Encodes texts with sentence-transformers, in its own process.

    Run by utils.embeddings (never run this in the main thread): keeping torch
    and transformers out of the app process avoids Streamlit's file watcher
    importing every transformers submodule, and frees the memory as soon as the 
    encoding is done.

    Parameters: 
        see main() description in this file.
    Usage: 
        $ python -m utils.embeddings_worker texts.json vectors.npy
    Prints: 
        LOADING / PROGRESS <done> <total> / DONE  (read by the parent)
"""

##### IMPORTS ##### ------------------------------------------------------------
import json
import sys
import numpy as np

# Only import name
from utils.embeddings import MODEL_NAME
CHUNK = 256 # Maximum number of tokens for all-MiniLM-L6-v2 (default embedding model)


##### FUNCTIONS ##### ----------------------------------------------------------
def say(*parts):
    """
        Prints status messages immediately, so another process or a user can see
        the progress even if there are other progress bar/prints in place.

        Parameters
        ----------
        *parts: any number of positional arguments.

        Returns
        -------
        The number of positional arguments, seperated by spaces.
    """
    print(*parts, flush=True)

def main(texts_path: str, out_path: str):
    """
        Loads text data, generate embeddings in batches, saves resulting vectors 
        to a NumPy file.

        Parameters
        ----------
        texts_path: str
            A JSON of string.
        out_path: str
            NumPy file path.
    """

    from sentence_transformers import SentenceTransformer

    say("LOADING")
    model = SentenceTransformer(MODEL_NAME)
    with open(texts_path, encoding="utf-8") as f:
        texts = json.load(f)

    batches = []
    for start in range(0, len(texts), CHUNK):
        say("PROGRESS", start, len(texts))
        batches.append(model.encode(
            texts[start:start + CHUNK], 
            batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True))
    np.save(out_path, np.vstack(batches).astype(np.float32))
    say("DONE")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
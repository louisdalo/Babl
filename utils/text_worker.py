"""
    Runs the local language models used by utils.masse_critique, in their own
    process (like utils.embeddings_worker: torch and transformers stay out of
    the Streamlit process, and the memory is freed when the work is done).

    Tasks
    -----
    translate : French -> English with MarianMT (Helsinki-NLP/opus-mt-fr-en).
        in : {"texts": [str, ...]}            out : {"texts": [str, ...]}
        Texts are cut into sentences (the model was trained on sentences), the
        sentences of all the texts are sorted by length and translated in
        batches (little padding), then put back together.
    match : closest subject of a vocabulary for each French tag, by meaning
        (multilingual sentence model).
        in : {"queries": [str], "vocab": [str], "cache": path or null}
        out : {"best": [[index in vocab, cosine similarity], ...]}

    Usage:
        $ python -m utils.text_worker translate in.json out.json
    Prints:
        LOADING / PROGRESS <done> <total> / DONE  (read by the parent)
    First run: downloads the models (about 300 MB and 470 MB).
"""

##### IMPORTS ##### ------------------------------------------------------------
import re
import sys
import json
import hashlib
from pathlib import Path
import numpy as np


##### PARAMETERS ##### ---------------------------------------------------------
MT_MODEL = "Helsinki-NLP/opus-mt-fr-en"
MATCH_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH = 32                  # sentences translated at once
NUM_BEAMS = 4               # translation quality / speed (1 = fastest)
MAX_TOKENS = 256            # per sentence
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…”\"»])\s+")


##### FUNCTIONS ##### ----------------------------------------------------------
def say(*parts):
    print(*parts, flush=True)


def split_sentences(text: str):
    """The sentences of a text (the whole text if it cannot be split, [] if it is empty)."""
    parts = [s.strip() for s in SENTENCE_SPLIT.split(text or "") if s.strip()]
    return parts


def translate_texts(texts, translate_batch, batch_size=BATCH, report=lambda done, total: None):
    """
        Translates texts sentence by sentence, in batches of similar length.

        Parameters
        ----------
        texts: list[str]
        translate_batch: function
            list[str] (sentences) -> list[str], same order.
        batch_size: int
        report: function
            Called with (sentences done, sentences total).

        Returns
        -------
        A list of translated texts, in the order of `texts`.
    """
    pieces = [(i, s) for i, t in enumerate(texts) for s in split_sentences(t)]
    order = sorted(range(len(pieces)), key=lambda k: len(pieces[k][1]))
    done = [""] * len(pieces)
    for start in range(0, len(order), batch_size):
        report(start, len(order))
        idx = order[start:start + batch_size]
        for k, out in zip(idx, translate_batch([pieces[k][1] for k in idx])):
            done[k] = out
    report(len(order), len(order))
    result = [[] for _ in texts]
    for (i, _), out in zip(pieces, done):
        result[i].append(out)
    return [" ".join(r) for r in result]


def best_matches(queries, vocab, encode, cache=None, chunk=1024, report=lambda done, total: None):
    """
        For each query, the closest vocabulary entry by cosine similarity.

        Parameters
        ----------
        encode: function
            list[str] -> array of unit-length vectors.
        cache: str or Path or None
            .npz file keeping the vectors of the vocabulary (the slow part),
            reused while the vocabulary and the model stay the same.

        Returns
        -------
        A list of [index in vocab, similarity].
    """
    key = hashlib.md5((MATCH_MODEL + "\n".join(vocab)).encode()).hexdigest()
    V = None
    if cache and Path(cache).exists():
        try:
            with np.load(cache, allow_pickle=False) as z:
                if str(z["key"]) == key:
                    V = z["emb"]
        except Exception:           # unreadable cache: recompute
            V = None
    if V is None:
        parts = []
        for start in range(0, len(vocab), chunk):
            report(start, len(vocab))
            parts.append(np.asarray(encode(vocab[start:start + chunk]), dtype=np.float32))
        V = np.vstack(parts)
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, key=np.array(key), emb=V)
    Q = np.asarray(encode(queries), dtype=np.float32)
    sims = Q @ V.T
    best = sims.argmax(axis=1)
    return [[int(j), float(sims[i, j])] for i, j in enumerate(best)]


def _translator():
    import torch
    # opus-mt-fr-en is a Marian model: the concrete classes are used instead of the Auto*
    # ones, whose model-type mappings changed between transformers versions.
    from transformers import MarianMTModel, MarianTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MarianTokenizer.from_pretrained(MT_MODEL)
    model = MarianMTModel.from_pretrained(MT_MODEL).to(device).eval()

    def translate_batch(sentences):
        enc = tok(sentences, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_TOKENS).to(device)
        with torch.inference_mode():
            out = model.generate(**enc, num_beams=NUM_BEAMS, max_new_tokens=MAX_TOKENS)
        return tok.batch_decode(out, skip_special_tokens=True)
    return translate_batch


def main(task: str, src: str, dst: str):
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    report = lambda done, total: say("PROGRESS", done, total)

    say("LOADING")
    if task == "translate":
        out = {"texts": translate_texts(data["texts"], _translator(), report=report)}
    elif task == "match":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MATCH_MODEL)
        encode = lambda texts: model.encode(texts, batch_size=64, normalize_embeddings=True,
                                            show_progress_bar=False, convert_to_numpy=True)
        out = {"best": best_matches(data["queries"], data["vocab"], encode, data.get("cache"), report=report)}
    else:
        sys.exit(f"Unknown task: {task!r}")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    say("DONE")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
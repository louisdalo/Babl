"""A set of function to build the features and model for recommendations."""

##### IMPORTS ##### ------------------------------------------------------------
import json
import math
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from utils.embeddings import MODEL_NAME as EMBEDDING_MODEL, EmbeddingsUnavailable, embed_books
from utils.series import series_groups
from utils.subjects import subject_key


##### PARAMETERS ##### ---------------------------------------------------------
# Relative importance of each block of features
W_TEXT, W_SUBJECTS, W_AUTHORS, W_NUMERIC, W_EMB = 1.0, 1.0, 0.5, 0.2, 1.0
ALL_BLOCKS = ("text", "subjects", "authors", "numeric", "embeddings")

C = 5.0                 # logistic regression: higher = less regularisation
UNREAD_WEIGHT = 0.7     # "interested / not for me" on an unread book is a weaker signal
NEUTRAL_WEIGHT = 0.5    # neutral books pull the prediction towards 50%
POS_THRESHOLD, NEG_THRESHOLD = 0.75, 0.25
CV_FOLDS, CV_REPEATS = 9, 3
SERIES_POWER = 1.0      # labelled books of one series share a total weight of 1 (1/n each); 0 = off
BASELINE = 0.5          # balanced accuracy of any constant prediction ("always like")
RATING_PRIOR_K = 10     # shrinkage strength: a book needs ~10 ratings before its own average counts as much as the prior

HISTORY_COLS = ["trained_at", "n_labels", "labels_hash", "cv_balanced_accuracy",
                "cv_std", "train_balanced_accuracy", "auc"]

SUBJECT_ALIASES = {
    "novel": "fiction", "novels": "fiction", "general literature": "fiction",
    "sf": "science fiction", "sci fi": "science fiction",
    "detective": "detective and mystery stories",
}

class NotEnoughData(ValueError):
    """Custom error raised when the labels cannot train a model."""


##### FEATURES ##### -----------------------------------------------------------
# Initialisation of a "Features" class, containing the skeleton of what the 
# model datasets will look like
@dataclass
class Features:
    ids: list                                           # book ids, same order as the rows of X
    index: dict                                         # book id -> row
    X: sp.csr_matrix                                    # actual features (e.g. subjects, authors, ratings, ...)
    names: list                                         # name of each column
    popularity: np.ndarray                              # Number of book ratings
    n_content: int                                      # columns that are not numeric (used for explanations)
    transformers: dict = field(default_factory=dict)    # fitted vectorizers / scaler (saved with the model)
    emb: np.ndarray | None = None                       # unit-length sentence embeddings (None if unavailable)
    embedding_error: str | None = None                  # why embeddings are missing, if they are
    blocks: tuple = ()                                  # blocks of features actually used
    groups: np.ndarray | None = None                    # series group of each book (books alone in their series: own group)

def _identity(names):
    """
        Return the input unchanged. This is used as a "pass-through" text 
        analyzer (see usage in build_features()).

        Parameters
        ----------
        names: object
            The value passed through the function unchanged. 

        Returns
        -------
        The same value as the one supplied.
    """
    return names

def _subject_analyzer(subjects):
    """
        Normalises a collection of subjects for TF-IDF processing.

        Each subject is converted to a normalised key, empty keys are discarded.

        Parameters
        ----------
        subjects: iterable
            Strings to normalise.

        Returns
        -------
        A list of normalised subject keys.
    """
    return [k for k in (subject_key(s) for s in subjects) if k]

def _tfidf(docs, prefix: str, weight: float, **kwargs):
    """
        Fits a TF-IDF vectorizer and return its sparse feature matrix.
        FYI: TF-IDF = Term Frequency-Inverse Document Frequency

        The resulting matrix is multiplied by given weight and each generated
        feature name is prefixed with given prefix. 
        
        If the corpus has an empty vocabulary, an empty sparse matrix is 
        returned instead.

        Parameters
        ----------
        docs: iterable
            Documents or token collections to vectorise.
        prefix: str
            Prefix added before every feature name.
        weight: float
            Multiplier applied to the TF-IDF matrix.
        **kwargs
            Additional keyword arguments, forwarded to TfidfVectorizer.

        Returns
        -------
        A tuple (X, names, vec):
            - X is a CSR (compressed sparse row) matrix,
            - names is the list of prefixed feature names,
            - vec is the fitted vectorizer.
    """
    try:
        vec = TfidfVectorizer(dtype=np.float32, **kwargs)
        X = vec.fit_transform(docs) * weight
        names = [f"{prefix}: {n}" for n in vec.get_feature_names_out()]
    except ValueError:      # empty vocabulary (tiny corpus)
        vec, X, names = None, sp.csr_matrix((len(docs), 0), dtype=np.float32), []
    return X.tocsr(), names, vec

def _author_names(book):
    """
        Extracts and normalises the author names for a book.

        Prefers the names in the book's authors list. If none are available,
        falls back to splitting the author string based on commas.

        Parameters
        ----------
        book: dict
            A book's metadata.

        Returns
        -------
        A list of strings. Non-empty author names stripped of surrounding 
        whitespace and converted to lowercase.
    """
    names = [a.get("name") for a in book.get("authors") or [] if isinstance(a, dict)]
    if not any(names) and book.get("author"):
        names = [n.strip() for n in book["author"].split(",")]
    return [n.casefold() for n in names if n]

def _global_rating_mean(books):
    """
        Mean of avg_rating over the books that actually have ratings.
        Used as the neutral value that shrunk ratings are pulled towards.
        Falls back to 3.5 if no book has a rating.

        Parameters
        ----------
        books: dict
            Dictitionary containing all books metadatas 
            (see utils.data.load_books()).

        Returns
        -------
        A book's rating (float)
    """
    vals = [b["avg_rating"] for b in books
            if b.get("avg_rating") is not None and (b.get("n_ratings") or 0) > 0]
    return float(np.mean(vals)) if vals else 3.5

def _numeric_raw(books, global_mean, k):
    """
        Builds the raw (unscaled) numeric matrix: year, log pages and shrunk
        average rating. Shared by _numeric_block() (training) and featurize()
        (new books) so both always compute the same columns.

        shrunk = (n * avg + k * global_mean) / (n + k)
        Books without ratings get global_mean (a neutral value).

        Parameters
        ----------
        books: dict
            Dictitionary containing all books metadatas 
            (see utils.data.load_books()).
        global_mean: float
            The output of _global_rating_mean().
        k: integer
            The strength of the shrinkage.

        Returns
        -------
        A tuple (array, column names). Missing year / length are NaN.
    """
    def log(v):
        return math.log1p(v) if v else None

    def shrunk_rating(b):
        avg, n = b.get("avg_rating"), b.get("n_ratings") or 0
        if avg is None or n <= 0:
            return global_mean
        return (n * avg + k * global_mean) / (n + k)

    cols = {
        "year": [b.get("first_publish_year") for b in books],
        "length": [log(b.get("pages")) for b in books],
        "avg rating": [shrunk_rating(b) for b in books],
    }
    arr = np.array([[np.nan if v is None else float(v) for v in col] for col in cols.values()]).T
    return arr, list(cols)

def _numeric_block(books):
    """
        Builds and scales the numeric feature block for a collection of books.

        Uses publication year, log-transformed page count and a shrunk average
        rating (popularity is deliberately NOT a feature: it measures exposure,
        not quality). Missing values are filled with per-column medians, then
        the columns are standardized, clipped, and weighted.

        Parameters
        ----------
        books: dict
            Dictitionary containing all books metadatas 
            (see utils.data.load_books()).
        
        Returns
        -------
        A tuple (X, names, metadata):
            - X is a float32 CSR (compressed sparse row) matrix,
            - names lists the numeric feature names,
            - metadata contains the fitted scaler, imputation medians, and the
              rating prior (global mean and k), needed to featurize new books.
    """
    global_mean = _global_rating_mean(books)
    arr, names = _numeric_raw(books, global_mean, RATING_PRIOR_K)
    med = np.nan_to_num(np.nanmedian(arr, axis=0))
    rows, columns = np.where(np.isnan(arr))
    arr[rows, columns] = med[columns]
    scaler = StandardScaler()
    arr = np.clip(scaler.fit_transform(arr), -3, 3) * W_NUMERIC
    return (sp.csr_matrix(arr.astype(np.float32)), names,
            {"scaler": scaler, "medians": med,
             "global_mean": global_mean, "rating_prior_k": RATING_PRIOR_K})

def _LEGACY_numeric_block(books):
    """
        Builds and scales the numeric feature block for a collection of books.

        Uses publication year, log-transformed page count, average rating, and
        log-transformed popularity. Missing values are filled with per-column
        medians, then the columns are standardized, clipped, and weighted.

        Parameters
        ----------
        books: dict
            Dictitionary containing all books metadatas 
            (see utils.data.load_books()).

        Returns
        -------
        A tuple (X, names, metadata):
            - X is a float32 CSR (compressed sparse row) matrix,
            - names lists the numeric feature names,
            - metadata contains the fitted scaler and imputation medians.
    """
    def log(v):
        """
            A simple function to log-transform a value.

            Parameters
            ----------
            v: number or None
                The value to transform. Should be a positive numeric.

            Returns
            -------
            float (log(1 + v)) or None (if v is 0 or negative).
        """
        return math.log1p(v) if v else None
    cols = {
        "year": [b.get("first_publish_year") for b in books],
        "length": [log(b.get("pages")) for b in books],
        "avg rating": [b.get("avg_rating") for b in books],
        "popularity": [log(b.get("popularity")) for b in books],
    }
    arr = np.array([[np.nan if v is None else float(v) for v in col] for col in cols.values()]).T
    med = np.nan_to_num(np.nanmedian(arr, axis=0))
    rows, columns = np.where(np.isnan(arr))
    arr[rows, columns] = med[columns]
    scaler = StandardScaler()
    arr = np.clip(scaler.fit_transform(arr), -3, 3) * W_NUMERIC
    return sp.csr_matrix(arr.astype(np.float32)), list(cols), {"scaler": scaler, "medians": med}

def build_features(books, blocks=ALL_BLOCKS, emb=None, progress=None):
    """
        Preprocesses the books into a feature matrix.

        The "blocks" parameter selects which families of features are used:
            - text:         TF-IDF of title + subtitle + summary (words and word pairs)
            - subjects:     TF-IDF of the (merged) subjects
            - authors:      TF-IDF of the authors
            - numeric:      scaled year, length, average rating, popularity
            - embeddings:   sentence embeddings of title + subtitle + summary

        Parameters
        ----------
        books: dict
            Dictitionary containing all books metadatas 
            (see utils.data.load_books()).
        blocks: iterable[str], optional
            Feature families to include. Defaults to ALL_BLOCKS.
        emb: numpy.ndarray or None, optional
            Precomputed book embeddings, or None if they need to be computed.
        progress: function
            A callback (progress bar) accepting (fraction, message) updates.

        Returns
        -------
        A completed Features instance.
    """
    report = progress or (lambda fraction, message: None)
    ids = list(books)
    items = [books[i] for i in ids]
    none_block = (sp.csr_matrix((len(items), 0), dtype=np.float32), [], None)

    report(0.0, "Preprocessing: summary words...")
    if "text" in blocks:
        docs = [" ".join(filter(None, [b.get("title"), b.get("subtitle"), b.get("summary")])) for b in items]
        X_text, n_text, v_text = _tfidf(docs, "word", W_TEXT, stop_words="english", ngram_range=(1, 2),
                                        min_df=3, max_df=0.5, max_features=8000, sublinear_tf=True)
    else:
        X_text, n_text, v_text = none_block
    report(0.25, "Preprocessing: subjects...")
    if "subjects" in blocks:
        X_subj, n_subj, v_subj = _tfidf([b.get("subjects") or [] for b in items], "subject", W_SUBJECTS,
                                        analyzer=_subject_analyzer, min_df=2)
    else:
        X_subj, n_subj, v_subj = none_block
    report(0.33, "Preprocessing: authors...")
    if "authors" in blocks:
        X_auth, n_auth, v_auth = _tfidf([_author_names(b) for b in items], "author", W_AUTHORS,
                                        analyzer=_identity, min_df=2)
    else:
        X_auth, n_auth, v_auth = none_block
    report(0.38, "Preprocessing: year, length, ratings...")
    X_num, n_num, num_tf = _numeric_block(items) if "numeric" in blocks else none_block

    X_emb, n_emb, emb_error, used = none_block[0], [], None, [b for b in blocks if b != "embeddings"]
    if "embeddings" in blocks:
        try:
            emb = emb if emb is not None else embed_books(
                books, progress=lambda f, m: report(0.42 + 0.48 * f, m))
            X_emb = sp.csr_matrix(emb * W_EMB)
            n_emb = [f"embedding: dim {k}" for k in range(emb.shape[1])]
            used.append("embeddings")
        except EmbeddingsUnavailable as exc:
            emb, emb_error = None, str(exc)
    else:
        emb = None

    report(0.92, "Detecting series...")
    groups = np.array(series_groups(books))
    report(0.97, "Assembling features...")
    X = sp.hstack([X_text, X_subj, X_auth, X_num, X_emb]).tocsr()     # embeddings last: not used in "why"
    return Features(
        ids=ids, index={b: i for i, b in enumerate(ids)}, X=X,
        names=n_text + n_subj + n_auth + n_num + n_emb,
        popularity=np.array([b.get("popularity") or 0 for b in items]),
        n_content=len(n_text) + len(n_subj) + len(n_auth),
        transformers={"text": v_text, "subjects": v_subj, "authors": v_auth, "numeric": num_tf,
                      "embedding_model": EMBEDDING_MODEL if emb is not None else None,
                      "weights": {"text": W_TEXT, "subjects": W_SUBJECTS, "authors": W_AUTHORS,
                                  "numeric": W_NUMERIC, "embeddings": W_EMB}},
        emb=emb, embedding_error=emb_error, blocks=tuple(used),
        groups=groups,
    )


##### TRAINING ##### -----------------------------------------------------------
# Initialisation of a "Result" class, containing the trained model and resutls
@dataclass
class Result:
    model: LogisticRegression       # Initial model instance
    n_pos: int                      # Number of positive labels
    n_neg: int                      # Number of negative labels
    n_neutral: int                  # Number of neutral labels
    train_bal_acc: float            # balanced accuracy on the books the model learned from
    proba: np.ndarray               # P(like) for every book
    importance: pd.DataFrame        # feature, name, block, coef, influence
    cv_bal_acc: float | None = None # cross-validated balanced accuracy (None: too few labels)
    cv_std: float | None = None     # cross-validated balanced accuracy std
    cv_scores: list | None = None   # one balanced accuracy per CV split
    auc: float | None = None        # ROC AUC on out-of-fold predictions
    roc: dict | None = None         # {"fpr": [...], "tpr": [...]}
    confusion: list | None = None   # [[TN, FP], [FN, TP]] on out-of-fold predictions
    oof: pd.DataFrame | None = None # out-of-fold P(like) and true class of each labelled book

def _block(name: str):
    """
        Maps a feature name to its human-readable feature block.

        Parameters
        ----------
        name: str
            A feature's name, typically prefixed with a block identifier.
            E.g. "word: ...".

        Returns
        -------
        A string representing the feature's block's name.
    """
    prefix = name.split(": ", 1)[0]
    return {"word": "summary words", "subject": "subjects", "author": "authors",
            "embedding": "sentence embeddings"}.get(prefix, "numeric")

def _fit(X, y, w, c=C):
    """
        Fits a logistic regression model, balanced by the weight of each class.

        Parameters
        ----------
        X: array-like or sparse matrix
            Training feature matrix.
        y: array-like
            Binary training labels.
        w: array-like
            Sample weights.
        c: float, optional
            Logistic regression regularization parameter C 
            (see LogisticRegression()).

        Returns
        -------
        The fitted logistic regression estimator.
    """
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
    model.fit(X, y, sample_weight=w)
    return model

def _design(X, d_idx, d_y, d_w, n_idx, n_w):
    """
        Assembles the weighted training matrix, labels, and sample weights.

        Each neutral book is included twice: once as a positive example and once as a negative example, with each
        copy down-weighted by NEUTRAL_WEIGHT.

        Parameters
        ----------
        X: sparse matrix
            Feature matrix indexed by book row.
        d_idx: array-like
            Row indices for decisive (positive or negative) labels.
        d_y: array-like
            Binary labels corresponding to ``d_idx``.
        d_w: array-like
            Sample weights for decisive labels.
        n_idx: array-like
            Row indices for neutral books.
        n_w: array-like
            Sample weights for neutral books before neutral weighting.

        Returns
        -------
        A tuple (X_train, y_train, weights):
            - the assembled CSR (compressed sparse row) feature matrix,
            - concatenated binary labels,
            - concatenated sample weights.
    """
    """Training set = decisive labels + neutral books (once as like, once as dislike, half weight)."""
    Xs, ys, ws = [X[d_idx]], [d_y], [d_w]
    if len(n_idx):
        Xn = X[n_idx]
        Xs += [Xn, Xn]
        ys += [np.ones(len(n_idx)), np.zeros(len(n_idx))]
        ws += [n_w * NEUTRAL_WEIGHT] * 2
    return sp.vstack(Xs).tocsr(), np.concatenate(ys), np.concatenate(ws)

def _importance(feats, model):
    """
        Computes a model's features coefficients and empirical influence.

        Influence measures contribution magnitude, while the coefficient
        sign indicates whether a feature pushes towards like or dislike.

        Parameters
        ----------
        feats: Features instance
            Feature data and feature names used by the trained model.
        model: LogisticRegression instance
            Fitted model whose coefficients are analyzed.

        Returns
        -------
        A pd.DataFrame with one row per feature and 5 columns: "feature", 
        "name", "block", "coef" and "influence".
    """
    coef = model.coef_[0]
    influence = np.asarray(abs(feats.X @ sp.diags(coef)).mean(axis=0)).ravel()
    return pd.DataFrame({
        "feature": feats.names,
        "name": [n.split(": ", 1)[-1] for n in feats.names],
        "block": [_block(n) for n in feats.names],
        "coef": coef,
        "influence": influence,
    })

def train_model(feats, label_rows, c=C, progress=None):
    """
        Trains a logistic regression model from user book labels.

        Labels for books in the same series are reweighted so that a large 
        series cannot dominate predictions. When enough distinct series are
        available, performance is evaluated with repeated stratified,
        group-aware cross-validation to keep a series within one split.

        Parameters
        ----------
        feats: Features instance
            Preprocessed book features, including series groups and row lookup.
        label_rows: dict
            Label records with 3 fields: "book_id", "label", and "read".
            The latest record for each book ID is used.
        c: float, optional
            Logistic regression regularization parameter C.
            (see LogisticRegression()).
        progress: function
            A callback (progress bar) accepting (fraction, message) updates.

        Returns
        -------
        A completed Result instance

        Raises
        ------
        NotEnoughData if there is only one type of label.
    """
    report = progress or (lambda fraction, message: None)
    report(0.0, "Preparing labels...")
    last = {r["book_id"]: r for r in label_rows}        # latest label per book
    pos, neg, neu = [], [], []
    for bid, r in last.items():
        i = feats.index.get(bid)
        if i is None:
            continue
        label, w = float(r["label"]), (1.0 if int(r["read"]) else UNREAD_WEIGHT)
        (pos if label >= POS_THRESHOLD else neg if label <= NEG_THRESHOLD else neu).append((i, w))
    if not pos or not neg:
        raise NotEnoughData("The model needs at least one book you like AND one you dislike.")

    series_size = Counter(feats.groups[i] for i, _ in pos + neg + neu)
    
    def weight(i, w):
        """
            Computes the effective sample weight for a labelled book.

            Applies the supplied label weight and adjusts it according to the
            size of the book's series group.

            Parameters
            ----------
            i: int
                Row index of the book in the feature data.
            w: float
                Base weight associated with the label.

            Returns
            -------
            A float, the base weight multiplied by the series-size adjustment.
        """
        return w * series_size[feats.groups[i]] ** -SERIES_POWER

    d_idx = np.array([i for i, _ in pos + neg])
    d_w = np.array([weight(i, w) for i, w in pos + neg])
    d_y = np.array([1] * len(pos) + [0] * len(neg))
    d_g = feats.groups[d_idx]
    n_idx = np.array([i for i, _ in neu], dtype=int)
    n_w = np.array([weight(i, w) for i, w in neu])
    n_g = feats.groups[n_idx]

    extra = {}
    k = min(CV_FOLDS, len(set(d_g[d_y == 1])), len(set(d_g[d_y == 0])))
    if k >= 2:
        scores, done, total = [], 0, k * CV_REPEATS
        oof_sum, oof_n = np.zeros(len(d_y)), np.zeros(len(d_y))
        for seed in range(CV_REPEATS):
            splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            for tr, te in splitter.split(np.zeros(len(d_y)), d_y, groups=d_g):
                report(0.05 + 0.80 * done / total, f"Cross-validation: split {done + 1} / {total}")
                done += 1
                if len(set(d_y[te])) < 2 or len(set(d_y[tr])) < 2:
                    continue
                keep = ~np.isin(n_g, d_g[te])       # neutral books of a test series stay out of training
                Xtr, ytr, wtr = _design(feats.X, d_idx[tr], d_y[tr], d_w[tr], n_idx[keep], n_w[keep])
                model = _fit(Xtr, ytr, wtr, c)
                p = model.predict_proba(feats.X[d_idx[te]])[:, 1]
                scores.append(float(balanced_accuracy_score(d_y[te], p >= 0.5)))
                oof_sum[te] += p
                oof_n[te] += 1
        ok = oof_n > 0
        if scores and len(set(d_y[ok])) == 2:
            oof, y_ok = oof_sum[ok] / oof_n[ok], d_y[ok]
            fpr, tpr, _ = roc_curve(y_ok, oof)
            extra = dict(
                cv_bal_acc=float(np.mean(scores)), cv_std=float(np.std(scores)), cv_scores=scores,
                auc=float(roc_auc_score(y_ok, oof)),
                roc={"fpr": fpr.tolist(), "tpr": tpr.tolist()},
                confusion=confusion_matrix(y_ok, oof >= 0.5, labels=[0, 1]).tolist(),
                oof=pd.DataFrame({"book_id": [feats.ids[i] for i in d_idx[ok]], "proba": oof,
                                  "actual": np.where(y_ok == 1, "liked", "disliked")}),
            )

    report(0.88, "Fitting the final model...")
    X, y, w = _design(feats.X, d_idx, d_y, d_w, n_idx, n_w)
    model = _fit(X, y, w, c)
    report(0.95, "Computing feature importance...")
    return Result(
        model=model, n_pos=len(pos), n_neg=len(neg), n_neutral=len(neu),
        train_bal_acc=float(balanced_accuracy_score(d_y, model.predict(feats.X[d_idx]))),
        proba=model.predict_proba(feats.X)[:, 1],
        importance=_importance(feats, model),
        **extra,
    )


##### SAVING ##### -------------------------------------------------------------
def save_result(feats, result, label_rows, out_dir):
    """
        Saves the trained model, metrics, feature importance, and score history.
        Creates the output directory if needed. 

        Parameters
        ----------
        feats: Features instance
            Feature data, names, transformers, and configuration metadata.
        result: Result instance
            Trained model and its evaluation results.
        label_rows: dict
            Label records with 3 fields: "book_id", "label", and "read".
            The latest record for each book ID is used.
        out_dir: str or pathlib.Path
            Directory in which output files will be written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")
    n_labels = result.n_pos + result.n_neg + result.n_neutral
    digest = hashlib.md5(json.dumps(sorted(
        (r["book_id"], str(r["label"]), str(r["read"])) for r in label_rows)).encode()).hexdigest()[:12]

    metrics = {
        "trained_at": now, "n_labels": n_labels, "labels_hash": digest,
        "n_positive": result.n_pos, "n_negative": result.n_neg, "n_neutral": result.n_neutral,
        "cv_balanced_accuracy": result.cv_bal_acc, "cv_std": result.cv_std,
        "cv_split_scores": result.cv_scores,
        "train_balanced_accuracy": result.train_bal_acc,
        "baseline_balanced_accuracy": BASELINE,
        "auc": result.auc, "confusion_matrix [[TN, FP], [FN, TP]]": result.confusion,
        "settings": {"C": result.model.C, "blocks": list(feats.blocks),
                     "embedding_model": feats.transformers.get("embedding_model"),
                     "weights": feats.transformers["weights"], "unread_weight": UNREAD_WEIGHT,
                     "neutral_weight": NEUTRAL_WEIGHT, "cv_folds": CV_FOLDS, "cv_repeats": CV_REPEATS,
                     "series_power": SERIES_POWER, "cv_grouped_by_series": True},
    }
    joblib.dump({"model": result.model, "feature_names": feats.names,
                 "transformers": feats.transformers, "metrics": metrics}, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    result.importance.sort_values("influence", ascending=False).to_csv(
        out_dir / "feature_importance.csv", index=False)

    history = read_history(out_dir)
    if history.empty or history["labels_hash"].iloc[-1] != digest:      # no duplicate lines
        row = pd.DataFrame([{
            "trained_at": now, "n_labels": n_labels, "labels_hash": digest,
            "cv_balanced_accuracy": result.cv_bal_acc, "cv_std": result.cv_std,
            "train_balanced_accuracy": result.train_bal_acc, "auc": result.auc,
        }], columns=HISTORY_COLS)
        pd.concat([history, row], ignore_index=True).to_csv(out_dir / "history.csv", index=False)

def read_history(out_dir):
    """
        Read the model-performance history CSV generated by save_result().

        If the history file does not exist, returns an empty DataFrame with
        the expected history columns.

        Parameters
        ----------
        out_dir : str or pathlib.Path
            Directory containing "history.csv".

        Returns
        -------
        A pd.DataFrame containing the history records.
    """
    path = Path(out_dir) / "history.csv"
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLS)
    return pd.read_csv(path, dtype={"labels_hash": str})


##### RECOMMENDATION ##### -----------------------------------------------------
def recommend(
        feats, result, exclude_ids, top_n: int=10, min_popularity: int=0, 
        liked_ids=(), discovery: bool=False, rated_ids=()):
    """
        A function to recommend high-scoring books that have not been excluded.

        Ranks books by the model's predicted probability of being liked, filters 
        out excluded books and books below the popularity threshold, provides up 
        to three positive feature contributions as explanations.
        
        If embeddings and liked-book IDs are available, also identifies the most 
        similar liked book. In discovery mode, excludes series containing books
        that were already rated and keeps at most one recommendation per 
        remaining series.

        Parameters
        ----------
        feats: Features instance
            Book features, IDs, popularity values, series groups, and optional
            sentence embeddings.
        result: Result instance
            Trained model, predicted probabilities, and model coefficients.
        exclude_ids: collection
            Book IDs that must not be recommended, typically already labelled.
        top_n: int, optional
            Maximum number of recommendations to return. Defaults to 10.
        min_popularity: number, optional
            Minimum popularity required for a book. Defaults to 0.
        liked_ids: iterable, optional
            IDs of liked books used to find a similar book by embeddings.
        discovery: bool, optional
            If true, exclude series containing any ID in rated_ids and
            return at most one book per remaining series.
        rated_ids: iterable, optional
            Book IDs considered already rated in discovery mode.

        Returns
        -------
        A list of recommendations ordered by predicted "like" probability. 
        Each item contains X fields: "id", "proba", "reasons" and "similar_to"
        (last one is a dict with a book ID and similarity score, or None).
    """
    coef = result.model.coef_[0]
    liked_idx = [feats.index[b] for b in liked_ids if b in feats.index]
    seen_groups = {feats.groups[feats.index[b]] for b in rated_ids if b in feats.index} if discovery else set()
    out = []
    for i in np.argsort(-result.proba):
        bid = feats.ids[i]
        if bid in exclude_ids or feats.popularity[i] < min_popularity:
            continue
        if discovery and feats.groups[i] in seen_groups:
            continue
        seen_groups.add(feats.groups[i])
        contrib = feats.X[i].toarray().ravel()[:feats.n_content] * coef[:feats.n_content]
        top = np.argsort(-contrib)[:3]
        similar = None
        if feats.emb is not None and liked_idx:
            sims = feats.emb[liked_idx] @ feats.emb[i]
            j = int(np.argmax(sims))
            similar = {"id": feats.ids[liked_idx[j]], "similarity": float(sims[j])}
        out.append({
            "similar_to": similar,
            "id": bid,
            "proba": float(result.proba[i]),
            "reasons": [feats.names[j] for j in top if contrib[j] > 0],
        })
        if len(out) == top_n:
            break
    return out


##### SCORING NEW BOOKS ##### --------------------------------------------------
# For books that were NOT in the catalogue the model was trained on (e.g. the
# Masse Critique books): their features are built with the vectorizers and the
# scaler fitted at training time (saved in model.joblib), so that the columns
# match the model's coefficients.
# Translated tags that never match an Open Library subject literally ("novel" -> "fiction").
# Keys and values are subject_key() forms; a value that is not in the model's vocabulary is ignored.
def snap_subjects(subjects, vocab):
    """
        Maps subjects that come from another source (e.g. translated Babelio
        tags) onto the subjects the model was trained on.
 
        The subjects TF-IDF only knows the subjects of its vocabulary: any
        other one is silently ignored. So each subject is tried as it is, then
        through SUBJECT_ALIASES, then with " fiction" added ("horror" ->
        "horror fiction"), then as a plural / singular.
 
        Parameters
        ----------
        subjects: iterable[str]
            Subjects of one book.
        vocab: dict or set
            Keys of the subjects vectorizer (vectorizer.vocabulary_).
 
        Returns
        -------
        A tuple (keys, n_matched): the vocabulary keys found (no duplicates),
        and how many of the subjects were matched.
    """
    found, n_matched = [], 0
    for s in subjects:
        k = subject_key(s)
        if not k:
            continue
        candidates = [k, SUBJECT_ALIASES.get(k), k + " fiction", k + "s",
                      k[:-1] if k.endswith("s") else None, k[:-3] + "y" if k.endswith("ies") else None]
        hit = next((c for c in candidates if c and c in vocab), None)
        if hit:
            n_matched += 1
            if hit not in found:
                found.append(hit)
    return found, n_matched
 
def featurize(books, transformers, emb=None, cache_path=None, progress=None):
    """
        Builds the feature matrix of books with already-fitted transformers.
 
        Same blocks, same order and same weights as build_features(), but
        nothing is fitted again.
 
        Parameters
        ----------
        books: dict
            Book id -> book (same schema as books.json).
        transformers: dict
            The "transformers" entry of model.joblib.
        emb: numpy.ndarray or None
            Precomputed embeddings of the books (rows follow list(books)).
        cache_path: str or Path or None
            Where to cache the embeddings of these books when emb is None.
            WARNING: never the catalogue's cache (utils.embeddings.CACHE_PATH):
            embed_books() rewrites its cache with the books it is given.
        progress: function
            A callback accepting (fraction, message) updates.
 
        Returns
        -------
        A CSR (compressed sparse row) matrix, one row per book.
    """
    ids = list(books)
    items = [books[i] for i in ids]
    n, w = len(items), transformers["weights"]
    empty = sp.csr_matrix((n, 0), dtype=np.float32)
 
    def tfidf(vec, docs, weight):
        return (vec.transform(docs) * weight).tocsr() if vec is not None else empty
 
    docs = [" ".join(filter(None, [b.get("title"), b.get("subtitle"), b.get("summary")])) for b in items]
    X_text = tfidf(transformers["text"], docs, w["text"])
    vec_subj = transformers["subjects"]
    subjects = [snap_subjects(b.get("subjects") or [], vec_subj.vocabulary_)[0] if vec_subj is not None else []
                for b in items]
    X_subj = tfidf(vec_subj, subjects, w["subjects"])
    X_auth = tfidf(transformers["authors"], [_author_names(b) for b in items], w["authors"])
 
    X_num, num = empty, transformers["numeric"]
    if num is not None and n:
        if "global_mean" not in num:
            raise ValueError("The saved model was trained with an older numeric block: "
                             "retrain it from the Recommendation page.")
        arr, _ = _numeric_raw(items, num["global_mean"], num["rating_prior_k"])
        rows, columns = np.where(np.isnan(arr))
        arr[rows, columns] = num["medians"][columns]
        X_num = sp.csr_matrix((np.clip(num["scaler"].transform(arr), -3, 3) * w["numeric"]).astype(np.float32))
 
    X_emb = empty
    if transformers.get("embedding_model"):
        if emb is None:
            if cache_path is None:
                raise ValueError("The model uses sentence embeddings: give `emb` or a `cache_path`.")
            emb = embed_books(books, cache_path=cache_path, progress=progress)
        X_emb = sp.csr_matrix(emb * w["embeddings"])
    return sp.hstack([X_text, X_subj, X_auth, X_num, X_emb]).tocsr()
 
 
def score_new_books(books, model_dir, emb=None, cache_path=None, progress=None):
    """
        Predicts the probability of liking for books unknown to the model.
 
        Parameters
        ----------
        books: dict
            Book id -> book (same schema as books.json; "summary" in English).
        model_dir: str or Path
            Folder containing model.joblib (see utils.data.MODEL_DIR).
        emb, cache_path, progress:
            See featurize().
 
        Returns
        -------
        A dict book id -> {"proba": float, "reasons": [feature names],
        "subjects_known": int, "subjects_total": int}. "reasons" are the (up
        to 3) content features that push most towards "like", like in
        recommend(); "subjects_known" / "subjects_total" tell how many of the
        book's subjects were matched to the model's vocabulary.
 
        Raises
        ------
        ValueError if the features do not match the saved model.
    """
    if not books:
        return {}
    bundle = joblib.load(Path(model_dir) / "model.joblib")
    model, names = bundle["model"], bundle["feature_names"]
    X = featurize(books, bundle["transformers"], emb, cache_path, progress)
    if X.shape[1] != len(names):
        raise ValueError(f"The features ({X.shape[1]} columns) do not match the saved model "
                         f"({len(names)}): retrain it from the Recommendation page.")
    proba = model.predict_proba(X)[:, 1]
    coef = model.coef_[0]
    vec_subj = bundle["transformers"]["subjects"]
    n_content = sum(n.split(": ", 1)[0] in ("word", "subject", "author") for n in names)
    out = {}
    for k, bid in enumerate(books):
        contrib = X[k].toarray().ravel()[:n_content] * coef[:n_content]
        top = np.argsort(-contrib)[:3]
        n_subjects = len([x for x in books[bid].get("subjects") or [] if subject_key(x)])
        n_known = (snap_subjects(books[bid].get("subjects") or [], vec_subj.vocabulary_)[1]
                   if vec_subj is not None else 0)
        out[bid] = {"proba": float(proba[k]), "reasons": [names[j] for j in top if contrib[j] > 0],
                    "subjects_known": n_known, "subjects_total": n_subjects}
    return out
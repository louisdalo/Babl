"""A place where all hard coded path are grouped together, for convenience."""

##### IMPORTS ##### ------------------------------------------------------------
from pathlib import Path


##### PATHS ##### --------------------------------------------------------------
# Public Data
PATH_OLDP_DATA = Path("OLDP")
BOOKS_PATH = PATH_OLDP_DATA / Path("books.json") # Output file from prepare_data.py
CACHE_PATH = PATH_OLDP_DATA / Path("embeddings.npz")
BOOKS_META_PATH = BOOKS_PATH.with_name("books_meta.json")

# Personal Data
PATH_PERSONAL_DATA = Path("my_infos")
HANDCRAFTED_BOOKS_PATH = PATH_PERSONAL_DATA / Path("handcrafted_books.json")    # File similar to books.json but containing the books added manually
MODEL_DIR = PATH_PERSONAL_DATA / Path("model")                                  # saved model, metrics and history
LABELS_PATH = PATH_PERSONAL_DATA / Path("labels.csv")                           # File with labels obtained from rating.py
PATH_MC_EMBEDDINGS = PATH_PERSONAL_DATA / Path("embeddings_masse_critique.npz")

MAPPING_PATH = PATH_PERSONAL_DATA / Path("tag_mapping.json")                    # tag -> catalogue subject, with its score (to review)
OVERRIDES_PATH = PATH_PERSONAL_DATA / Path("tag_overrides.json")                # your own choices: {"tag": "Subject" or null}
VOCAB_CACHE_PATH = PATH_PERSONAL_DATA / Path("tag_vocab_embeddings.npz")
CATALOGUE_PATHS = (BOOKS_PATH, HANDCRAFTED_BOOKS_PATH)
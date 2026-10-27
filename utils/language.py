"""A set of function to identify the language of a text."""


##### IMPORTS ##### ------------------------------------------------------------
import re
import unicodedata
from collections import Counter
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder


##### PARAMETERS ##### ---------------------------------------------------------
# A sentence is dropped only if the detector is sure it is NOT English.
SENTENCE_MIN_CHARS = 25     # Short strings are too noisy to judge, so they are always kept.
FAST_PATH_CONF = 0.90       # if more than this percentage of a text is in English: keeps text.
SUBJECT_REJECT_CONF = 0.90  # If language is english and confidence is above this percentage: keep text.
TITLE_REJECT_CONF = 0.90    # If language is NOT english and confidence is above this percentage: remove text.

SENTENCE_SPLIT = re.compile(r"(?<=[.!?…”\"])\s+") # Splits a text into sentences.
WORD_RE = re.compile(r"[a-z]+")                   # Splits a sentence into words.

VOCAB_MIN_COUNT = 3         # Minimum number of times a word has to appear in the whole corpus of texts.
MIN_VOCAB_SIZE = 5000       # Minimum number of distinct words the final vocabulary must contain.

# The detector cannot judge titles (it calls "The Alchemist" German, 8% English),
# but small words give them away: "O Alquimista", "Le petit prince", "Der Process".
ENGLISH_CUES = frozenset(
    """the of and to in for with on at by from is are an how why what who my 
    your our his her their""".split())
FOREIGN_CUES = frozenset(
    """o os um uma el los las un una unos le les du des der die das dem den ein 
    eine het een il lo gli della dello dei delle del de da do dos di et und och 
    og y em na nos por para que qui sur dans pour avec sans con sin mit von zur 
    zum ou""".split())


##### DETECTOR ##### -----------------------------------------------------------
@lru_cache(maxsize=1)
def _detector():
    """
        Creates the language detection model. 
        Text can be any language so no selection is made.

        Returns
        ------
        A LanguageDetector object
    """
    return LanguageDetectorBuilder.from_all_languages().build()

def _top(text: str):
    """
        A helper function that detects which language the text is made from and
        returns only two metrics : primary language detected and confidence.

        Parameters
        ----------
        text: string
            A text in any language.

        Returns
        ------
        A tuple (language, confidence) of the most likely language of `text`.
    """
    best = _detector().compute_language_confidence_values(text)[0]
    return best.language, best.value

def is_latin_script(text: str):
    """
        A helper function that detects which if a text contains letters from a
        non-Latin alphabet, which means text is not (or not only) english.

        Parameters
        ----------
        text: string
            A text in any language.

        Returns
        ------
        A boolean
   """
    for ch in text:
        if ch.isalpha() and not unicodedata.name(ch, "").startswith("LATIN"):
            return False
    return True


##### TEXT (summary, first sentence, notes) ##### ------------------------------
def keep_english(text: str, min_chars: int=0):
    """
        Removes the non-English sentences from a text. Open Library works often
        have several blurbs glued together (English + German + Spanish...), so
        looking at the whole text is not enough.

        Parameters
        ----------
        text: string
            A text in any language.
        min_chars: int
            The minimum number of character needed to analyse a text.

        Returns
        ------
        Returns the English-only text, or "".
    """
    if not text:
        return ""
    # Fast path: the (vast) majority of texts are plainly English.
    if _detector().compute_language_confidence(text, Language.ENGLISH) >= FAST_PATH_CONF:
        return text if len(text) >= max(min_chars, 1) else ""

    kept, prev_ok = [], False
    for sentence in SENTENCE_SPLIT.split(text):
        if len(sentence) >= SENTENCE_MIN_CHARS:
            ok = _top(sentence)[0] == Language.ENGLISH
        else:   # too short to judge: follows its neighbour
            ok = prev_ok and is_latin_script(sentence)
        if ok:
            kept.append(sentence)
        prev_ok = ok
    out = " ".join(kept).strip()
    return out if len(out) >= max(min_chars, 1) else ""


##### SUBJECTS ##### -----------------------------------------------------------
def english_vocabulary(
        texts, min_count: int=VOCAB_MIN_COUNT, min_size: int=MIN_VOCAB_SIZE):
    """
        Builds a set of English words from texts that are already known to be
        English (the kept summaries). It is used to judge one-word subjects,
        because the language detector is useless on a single word (it gives
        "Fiction" only 15% English and "Novela" 2%).

        Parameters
        ----------
        texts: iterable[list]
            A list of texts (str).
        min_count: int
            The minimum number of times a word has to appear in texts.
        min_size: int
            Minimum number of distinct words the final vocabulary must contain.          

        Returns
        ------
        Returns a dict of words, or None if corpus is too small.
    """
    counts = Counter()
    for t in texts:
        counts.update(WORD_RE.findall(t.lower()))
    vocab = {w for w, n in counts.items() if n >= min_count}
    return vocab if len(vocab) >= min_size else None

_STEM_CACHE = {}
def _stems(vocab):
    """
       Builds a set of six-letter prefixes from a vocabulary and caches that set 
       so it doesn't have to rebuild it every time the function is called with 
       the same vocabulary object.

        Parameters
        ----------
        vocab: iterable[list]
            A list of words.        

        Returns
        ------
        A boolean
    """
    # prefix index, rebuilt only when the vocabulary object changes
    key = id(vocab)
    if _STEM_CACHE.get("key") != key:
        _STEM_CACHE["key"] = key
        _STEM_CACHE["stems"] = {v[:6] for v in vocab if len(v) >= 6}
    return _STEM_CACHE["stems"]

def _in_vocab(word: str, vocab):
    """
        Checks if a given word is in the given vocabulary.
        Word must have an exact match, plural/singular variants, or (for long 
        words) a shared 6-letter stem.

        Parameters
        ----------
        word : str
            A word.
        vocab: iterable[list]
            A list of words.        

        Returns
        ------
        A boolean
    """
    w = word.lower()
    if w in vocab or w.rstrip("s") in vocab or (w + "s") in vocab or (w + "es") in vocab:
        return True
    if w.endswith("ies") and (w[:-3] + "y") in vocab:
        return True
    if len(w) >= 9:         # "dystopias" ~ "dystopian", "biographies" ~ "biography"
        return w[:6] in _stems(vocab)
    return False

@lru_cache(maxsize=1)
def _subject_detector():
    """
        Builds a language detector object that only picks from the selected
        languages.
    """
    return LanguageDetectorBuilder.from_languages(
        Language.ENGLISH, Language.FRENCH, Language.SPANISH, Language.GERMAN,
        Language.DUTCH, Language.ITALIAN, Language.PORTUGUESE).build()

# cache the results of function calls (so that similar subjects are re-used, 
# instead of re-computing them every time)
@lru_cache(maxsize=200_000)
def _phrase_is_english(subject: str):
    """
        Checks whether a phrase is in english or not.

        Parameters
        ----------
        subject : str
            A sentence/samll text.     

        Returns
        ------
        A boolean
    """
    d = _subject_detector()
    if d.detect_language_of(subject) == Language.ENGLISH:
        return True
    return d.compute_language_confidence(subject, Language.ENGLISH) >= 0.35

def subject_is_english(subject: str, vocab=None):
    """
        Checks whether a subject is in english or not.

        Outputs True if a subject tag looks English. Depending on subjects, uses
        different methods:
        - several words (or hyphenated): language detector, which rejects
          "Novela estadounidense" but not "Self-help"
        - one word: must be a known English word (rejects "Novela", "Alquimia");
          without `vocab`, one-word tags are accepted
        - junk tags (underscores, quotes: "open_syllabus_project") are rejected
        
        Parameters
        ----------
        subject : str
            A sentence/small text.    
        vocab: iterable[list]
            A list of words.     

        Returns
        ------
        A boolean
    """
    if not is_latin_script(subject) or "_" in subject or '"' in subject:
        return False
    if len(re.split(r"[\s-]+", subject.strip())) == 1:
        return (vocab is None or _in_vocab(subject, vocab)) and _phrase_is_english(subject)
    return _phrase_is_english(subject)


##### TITLES ##### -------------------------------------------------------------
def _norm_title(title: str):
    """
        Normalizes a title into a simplified, consistent format so that titles 
        can be compared more easily, even if they contain accents, punctuation,
        or inconsistent spacing.

        Parameters
        ----------
        title : str
            A sentence/small text.       

        Returns
        ------
        A (cleaned) string.
    """
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-z0-9 ]+", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()

def _words(title: str):
    """
        Divides a string into words.

        Parameters
        ----------
        title : str
            A sentence/small text.       

        Returns
        ------
        A set (dict) of strings.
    """
    return set(_norm_title(title).split())

def _has_english_cue(title: str):
    """
        Checks if a title contains english cues.

        Parameters
        ----------
        title : str
            A sentence/small text.       

        Returns
        ------
        A boolean
    """
    return bool(_words(title) & ENGLISH_CUES) and not _words(title) & FOREIGN_CUES

def _has_foreign_cue(title: str):
    """
        Checks if a title contains foreign cues.

        Parameters
        ----------
        title : str
            A sentence/small text.       

        Returns
        ------
        A boolean
    """
    return bool(_words(title) & FOREIGN_CUES) and not _words(title) & ENGLISH_CUES

def english_title(work_title: str, edition_titles: Counter):
    """
        Picks the title to display for a work.

        The title stored on an Open Library "work" field can be in any language
        (e.g. "O Alquimista"), while its English editions carry the English 
        title (e.g. "The Alchemist"). The detector alone cannot catch "O Alquimista"
        (too short), so the work title is compared with the titles of the
        English editions.

        Parameters
        ----------
        work_title : str
            Title of the work (small text).
        edition_titles : Counter
            Title -> number of English editions using it.

        Returns
        -------
        The title to use, or None if no English title could be found.
    """
    work_title = (work_title or "").strip()
    total = sum(edition_titles.values())
    own = Counter()
    for t, n in edition_titles.items():
        own[_norm_title(t)] += n

    # 1) Clearly not English by itself (other script, or a long foreign title)
    foreign = not is_latin_script(work_title)
    if not foreign and len(work_title.split()) >= 2:
        lang, conf = _top(work_title)
        foreign = lang != Language.ENGLISH and conf >= TITLE_REJECT_CONF

    # 2) Rare among English editions while another title dominates
    if not foreign and total >= 3 and edition_titles:
        top_title, top_n = edition_titles.most_common(1)[0]
        share = own[_norm_title(work_title)] / total
        foreign = (share < 0.10 and top_n / total >= 0.30
                   and _norm_title(top_title) != _norm_title(work_title))

    # 3) Foreign small words ("O Alquimista") while English editions have a
    #    title with English small words ("The Alchemist"). With no such edition
    #    the title is kept: "Les Misérables" is also the English title.
    if not foreign and _has_foreign_cue(work_title):
        foreign = any(_has_english_cue(t) for t in edition_titles)

    if not foreign:
        return work_title

    # Replace by an English edition title (most common first): the ones with
    # English small words, then any Latin-script one that is not clearly foreign.
    wanted = _norm_title(work_title)
    ranked = [t.strip() for t, _ in edition_titles.most_common()
              if t.strip() and is_latin_script(t) and _norm_title(t) != wanted]
    for t in ranked:
        if _has_english_cue(t):
            return t
    for t in ranked:
        if _has_foreign_cue(t):
            continue
        lang, conf = _top(t) if len(t.split()) >= 2 else (Language.ENGLISH, 0)
        if lang == Language.ENGLISH or conf < TITLE_REJECT_CONF:
            return t
    return None
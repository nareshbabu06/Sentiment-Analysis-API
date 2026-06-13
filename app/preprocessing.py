import re
from typing import Iterable, List

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize

try:
    from nltk.corpus import stopwords
except ImportError:  # pragma: no cover
    stopwords = None


_FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
}
_NEGATION_WORDS = {"no", "not", "nor", "never"}


def _load_stopwords() -> set[str]:
    if stopwords is None:
        return _FALLBACK_STOPWORDS - _NEGATION_WORDS
    try:
        return set(stopwords.words("english")) - _NEGATION_WORDS
    except LookupError:
        return _FALLBACK_STOPWORDS - _NEGATION_WORDS


_STOPWORDS = _load_stopwords()
_LEMMATIZER = WordNetLemmatizer()


def _lemmatize(token: str) -> str:
    try:
        return _LEMMATIZER.lemmatize(token)
    except LookupError:
        return token


def preprocess_text(text: str) -> str:
    normalized = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = []
    for token in wordpunct_tokenize(normalized):
        if not token.isalpha() or token in _STOPWORDS:
            continue
        tokens.append(_lemmatize(token))
    return " ".join(tokens)


def preprocess_corpus(texts: Iterable[str]) -> List[str]:
    return [preprocess_text(text) for text in texts]

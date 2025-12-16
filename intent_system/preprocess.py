#preprocess
import re
from typing import Tuple

# Optional spaCy lemmatization
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except Exception:
    _nlp = None

WAKE_WORD = "lappy"

# Common filler / noise words (can be excluded)
FILLER_WORDS = {
    "uh", "um", "umm", "please", "the", "a", "an"
}


def _normalize_text(text: str) -> str:
    #Lowercase, remove punctuation, normalize whitespace
    
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)   # keep apostrophes
    text = re.sub(r"\s+", " ", text)
    return text


def _remove_wake_word(text: str, wake_word: str) -> Tuple[bool, str]:
    #Detects and removes wake word.
    #Returns (has_wake_word, cleaned_text)
    pattern = rf"\b{re.escape(wake_word)}\b"
    if not re.search(pattern, text):
        return False, text

    cleaned = re.sub(pattern, "", text, count=1)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return True, cleaned


def _remove_fillers(text: str) -> str:
    """
    Removes filler words
    """
    tokens = text.split()
    tokens = [t for t in tokens if t not in FILLER_WORDS]
    return " ".join(tokens)


def _lemmatize(text: str) -> str:
    #Lemmatize words using spaCy if available
    if not _nlp:
        return text

    doc = _nlp(text)
    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_space and not token.is_punct
    ]
    return " ".join(lemmas)


def preprocess_text(raw_text: str) -> Tuple[bool, str]:
    #Full preprocessing pipeline.
    #Returns:
    #has_wake_word (bool)
    #cleaned_text (str)
    if not raw_text or not raw_text.strip():
        return False, ""

    text = _normalize_text(raw_text)
    has_wake, text = _remove_wake_word(text, WAKE_WORD)

    text = _remove_fillers(text)
    text = _lemmatize(text)

    text = re.sub(r"\s+", " ", text).strip()
    return has_wake, text


# Quick self-test
if __name__ == "__main__":
    samples = [
        "Hey Lappy, can you please open YouTube",
        "lappy what was the time",
        "umm hi there",
        "Lappy I was running fast"
    ]

    for s in samples:
        has_wake, cleaned = preprocess_text(s)
        print(f"RAW: {s}")
        print(f"WAKE: {has_wake}")
        print(f"CLEANED: {cleaned}")
        print("-" * 40)

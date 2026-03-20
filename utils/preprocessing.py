"""
Text preprocessing utilities for the Legal Document Analysis System.
Handles tokenisation, lemmatisation, and stopword removal.
NLTK data is downloaded lazily (once) so imports are fast.
"""
import re
import nltk

# ── Lazy one-time NLTK downloads ──────────────────────────────────────────────
_NLTK_READY = False

def _ensure_nltk():
    global _NLTK_READY
    if _NLTK_READY:
        return
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)
    _NLTK_READY = True

# ── Legal-domain stopwords to keep (they carry meaning) ──────────────────────
LEGAL_KEEP = {
    "not", "no", "nor", "against", "above", "below", "between",
    "party", "parties", "agreement", "contract", "shall", "herein",
    "whereas", "therefore", "thereof", "thereto",
}

def preprocess_text(text: str) -> str:
    """
    Clean and normalise a legal document string.

    Steps:
      1. Lowercase
      2. Remove non-alphabetic characters (preserve spaces)
      3. Tokenise
      4. Remove stopwords (but keep legal-domain terms)
      5. Lemmatise

    Returns a single whitespace-joined string ready for TF-IDF.
    """
    _ensure_nltk()

    from nltk.corpus import stopwords
    from nltk.stem   import WordNetLemmatizer

    stop_words  = set(stopwords.words("english")) - LEGAL_KEEP
    lemmatizer  = WordNetLemmatizer()

    text   = text.lower()
    text   = re.sub(r"[^a-z\s]", " ", text)   # keep only letters + spaces
    text   = re.sub(r"\s+", " ", text).strip()

    tokens = nltk.word_tokenize(text)
    clean  = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 1
    ]
    return " ".join(clean)

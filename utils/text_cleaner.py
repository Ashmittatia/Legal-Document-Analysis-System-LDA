"""
text_cleaner.py
---------------
Lightweight surface-level cleaner (no NLP dependencies).
Use preprocess_text() from preprocessing.py for full NLP pipeline.
"""
import re
import string
import unicodedata

def clean_text(text: str) -> str:
    """
    Basic surface cleaning:
      - Normalise unicode to ASCII where possible
      - Collapse whitespace
      - Strip punctuation
      - Lowercase
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_boilerplate(text: str) -> str:
    """Remove common legal boilerplate phrases to reduce noise."""
    BOILERPLATE = [
        r"this agreement is entered into",
        r"witnesseth\s*:",
        r"in witness whereof",
        r"hereinafter referred to as",
        r"the parties agree as follows",
    ]
    for pattern in BOILERPLATE:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()

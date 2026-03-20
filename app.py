import os
import io
import logging

from flask import Flask, render_template, request, jsonify
import joblib
import spacy
import numpy as np

from utils.preprocessing import preprocess_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH      = os.path.join("model", "classifier.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")

try:
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logger.info("Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    logger.error("Could not load model files: %s", e)
    model = vectorizer = None

try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded.")
except OSError:
    logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

ENTITY_META = {
    "PERSON":   {"label": "Person",        "color": "#4f93ce"},
    "ORG":      {"label": "Organisation",  "color": "#e07b39"},
    "GPE":      {"label": "Location",      "color": "#6dab6d"},
    "DATE":     {"label": "Date",          "color": "#a77cbf"},
    "MONEY":    {"label": "Money",         "color": "#c9a227"},
    "LAW":      {"label": "Law / Statute", "color": "#d15f5f"},
    "TIME":     {"label": "Time",          "color": "#5fafaf"},
    "CARDINAL": {"label": "Number",        "color": "#8c8c8c"},
    "NORP":     {"label": "Nationality",   "color": "#c67bba"},
    "FAC":      {"label": "Facility",      "color": "#7cb8a0"},
    "EVENT":    {"label": "Event",         "color": "#d4875b"},
}

def _extract_text_from_pdf(file_storage) -> str:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_storage.read()))
        pages  = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception as e:
        logger.warning("PDF extraction failed: %s", e)
        return ""

def _analyse(text: str) -> dict:
    if not text.strip():
        return {"error": "Empty text provided."}
    if model is None or vectorizer is None:
        return {"error": "Model not loaded. Run train_model.py first."}

    preprocessed = preprocess_text(text)
    vectorized   = vectorizer.transform([preprocessed])
    prediction   = model.predict(vectorized)[0]

    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(vectorized)[0]
        confidence = float(np.max(proba))
        classes    = model.classes_
        top_idx    = np.argsort(proba)[::-1][:3]
        top3 = [
            {"label": classes[i], "prob": round(float(proba[i]) * 100, 1)}
            for i in top_idx
        ]
    else:
        confidence = None
        top3       = []

    entities = []
    if nlp:
        doc  = nlp(text[:100_000])
        seen = set()
        for ent in doc.ents:
            key = (ent.label_, ent.text)
            if key in seen:
                continue
            seen.add(key)
            meta = ENTITY_META.get(ent.label_, {"label": ent.label_, "color": "#aaaaaa"})
            entities.append({
                "label":    ent.label_,
                "friendly": meta["label"],
                "text":     ent.text,
                "color":    meta["color"],
            })

    return {
        "prediction":  prediction,
        "confidence":  round(confidence * 100, 1) if confidence is not None else None,
        "top3":        top3,
        "entities":    entities,
        "word_count":  len(text.split()),
        "char_count":  len(text),
    }

@app.route("/", methods=["GET", "POST"])
def index():
    result     = None
    error      = None
    input_text = ""

    if request.method == "POST":
        uploaded = request.files.get("document_file")
        if uploaded and uploaded.filename:
            if uploaded.filename.lower().endswith(".pdf"):
                input_text = _extract_text_from_pdf(uploaded)
                if not input_text:
                    error = "Could not extract text from the PDF. Try pasting the text manually."
            elif uploaded.filename.lower().endswith(".txt"):
                input_text = uploaded.read().decode("utf-8", errors="ignore")
            else:
                error = "Unsupported file type. Please upload a .txt or .pdf file."
        else:
            input_text = request.form.get("document_text", "").strip()

        if not error:
            result = _analyse(input_text)
            if "error" in result:
                error  = result["error"]
                result = None

    return render_template("index.html", result=result, error=error, input_text=input_text)

@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    return jsonify(_analyse(text))

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": model is not None, "spacy": nlp is not None})

if __name__ == "__main__":
    app.run(debug=True)

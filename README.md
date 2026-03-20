# ⚖ Legal Document Analysis System (LDA)

An NLP-powered web application that **classifies legal documents** and **extracts named entities** — with confidence scoring, PDF upload, and a JSON API.

Built with Python · Flask · spaCy · NLTK · scikit-learn

---

## ✅ Features

| Feature | Details |
|---|---|
| Document Classification | Identifies type (Contract, Will, IP, Compliance, Business Formation) |
| Confidence Score | Probability of predicted class + top-3 breakdown |
| Named Entity Recognition | Persons, Organisations, Dates, Locations, Laws, Money, and more |
| PDF & TXT Upload | Upload files directly — no copy-paste needed |
| JSON REST API | `POST /api/analyse` for programmatic access |
| Health Check | `GET /health` — model/spaCy load status |
| Modern UI | Dark terminal aesthetic, entity colour badges, confidence bar |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Train the model  (requires the CSV dataset)
python train_model.py

# 3. Run the web app
python app.py
# → http://127.0.0.1:5000
```

For production:
```bash
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

---

## 📁 Project Structure

```
LDA/
├── app.py                  # Flask app — classification + NER + routes
├── train_model.py          # Training script with CV, metrics, confusion matrix
├── requirements.txt        # Pinned dependencies
├── model/
│   ├── classifier.pkl      # Trained LogisticRegression
│   ├── vectorizer.pkl      # TF-IDF vectorizer
│   └── pipeline.pkl        # Full sklearn Pipeline (convenience)
├── templates/
│   └── index.html          # Dark-theme UI
└── utils/
    ├── preprocessing.py    # Tokenise · lemmatise · stopwords (NLTK)
    └── text_cleaner.py     # Surface cleaning · boilerplate removal
```

---

## 🔌 API Usage

```bash
curl -X POST http://localhost:5000/api/analyse \
     -H "Content-Type: application/json" \
     -d '{"text": "This Agreement is entered into by Acme Corp..."}'
```

Response:
```json
{
  "prediction": "Contracts",
  "confidence": 94.3,
  "top3": [
    {"label": "Contracts",                    "prob": 94.3},
    {"label": "Business Formation Documents", "prob":  3.1},
    {"label": "Legal Compliance Documents",   "prob":  1.8}
  ],
  "entities": [
    {"label": "ORG", "friendly": "Organisation", "text": "Acme Corp", "color": "#e07b39"}
  ],
  "word_count": 8,
  "char_count": 48
}
```

---

## 📊 Model Details

- **Algorithm**: Logistic Regression (L2, lbfgs solver)
- **Features**: TF-IDF up to bi-grams, `sublinear_tf=True`, `min_df=2`
- **Evaluation**: 5-fold stratified cross-validation + held-out test set
- **Training output**: accuracy, per-class F1, confusion matrix

---

## 🛠 Improvements Over v1

- Confidence scores + top-3 class probabilities
- PDF and .txt file upload support
- JSON REST API (`/api/analyse`) + health endpoint (`/health`)
- Legal-domain stopwords preserved during preprocessing
- Stratified train/test split + cross-validation in training
- Full classification report + confusion matrix printed during training
- Lazy NLTK downloads (no slowdown on import)
- `text_cleaner.py` now has useful boilerplate-removal utility
- Dark UI with colour-coded entity badges, confidence meter, char counter, spinner

---

## 👤 Author

**Ashmit Tatia** · ashmit789@gmail.com · [@Ashmittatia](https://github.com/Ashmittatia)

MIT License

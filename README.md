# ⚖️ LDA — Legal Document Analysis System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/spaCy-3.6-09A3D5?style=flat-square&logo=spacy&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLTK-3.8-154f3c?style=flat-square"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square"/>
</p>

<p align="center">
  An NLP-powered web application that <strong>classifies legal documents</strong> and <strong>extracts named entities</strong> —<br/>
  with confidence scoring, PDF upload support, and a JSON REST API.
</p>

---

## 📸 UI Preview

<img width="1197" height="586" alt="image" src="https://github.com/user-attachments/assets/ccfe0eca-e43a-436e-90c3-218e1c999676" />

&nbsp;
| Document Classification | Named Entity Extraction |
|------------------------|------------------------|
| Predicted category · confidence bar · top-3 breakdown | Colour-coded entity badges · type labels · deduplication |
&nbsp;
<img width="1168" height="688" alt="image" src="https://github.com/user-attachments/assets/35366449-fe3d-4ab2-939c-4655d8fe555c" />
<img width="1202" height="443" alt="image" src="https://github.com/user-attachments/assets/5c73b9a9-f4e4-4a92-8e87-e85697c178f7" />

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1 — Input Sources                                     │
│  Paste text  │  Upload .txt  │  Upload .pdf                 │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│  Layer 2 — Text Preprocessing                                │
│  Surface cleaning · Tokenisation · Lemmatisation · Stopwords │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│  Layer 3 — ML Pipeline                                       │
│  TF-IDF Vectorizer (bi-grams)  │  Logistic Regression (L2)  │
│  (sklearn Pipeline saved as pipeline.pkl)                   │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│  Layer 4 — Named Entity Recognition                          │
│  spaCy en_core_web_sm · 11 entity types · deduplication     │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│  Layer 5 — Flask Web Application                             │
│  Dark UI (index.html)  │  /api/analyse  │  /health          │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- **Document Classification** — identifies 5 legal document types: Contracts, Wills, IP Documents, Legal Compliance, and Business Formation Documents
- **Confidence Scoring** — probability of the predicted class + full top-3 breakdown
- **Named Entity Recognition** — 11 entity types (Persons, Organisations, Locations, Dates, Laws, Money, and more) rendered as colour-coded badges
- **File Upload** — drag-and-drop `.txt` or `.pdf` files; no copy-paste needed
- **JSON REST API** — `POST /api/analyse` for programmatic access
- **Health Endpoint** — `GET /health` reports model and spaCy load status
- **Dark Terminal UI** — entity colour badges, animated confidence bar, character counter, loading spinner

---

## 📁 Project Structure

```
LDA/
├── app.py                  # Flask app — classification, NER, all routes
├── train_model.py          # Training script — CV, metrics, confusion matrix
├── requirements.txt        # Pinned dependencies
├── model/
│   ├── classifier.pkl      # Trained LogisticRegression
│   ├── vectorizer.pkl      # TF-IDF vectorizer
│   └── pipeline.pkl        # Full sklearn Pipeline (convenience)
├── templates/
│   └── index.html          # Dark-theme frontend
└── utils/
    ├── preprocessing.py    # Tokenise · lemmatise · stopwords (NLTK)
    └── text_cleaner.py     # Surface cleaning · boilerplate removal
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### 1. Clone and install

```bash
git clone https://github.com/Ashmittatia/lda.git
cd lda
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Train the model

```bash
python train_model.py
```

Expected output:
```
[INFO] Loaded 1,500 samples across 5 classes.
[INFO] Running 5-fold stratified cross-validation...
       CV accuracy: 0.9413 ± 0.0121
[INFO] Test Accuracy: 0.9467  (94.7%)
[DONE] Training complete.
       model/classifier.pkl
       model/vectorizer.pkl
       model/pipeline.pkl
```

### 3. Run the web app

```bash
python app.py
# → http://127.0.0.1:5000
```

For production:

```bash
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

---

## 🔌 API Endpoints

Base URL: `http://localhost:5000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI — paste text or upload a file |
| `POST` | `/api/analyse` | JSON classification + NER response |
| `GET` | `/health` | Model and spaCy load status |

### Example request

```bash
curl -X POST http://localhost:5000/api/analyse \
     -H "Content-Type: application/json" \
     -d '{"text": "This Agreement is entered into by Acme Corp and Jane Doe on January 1, 2024."}'
```

### Example response

```json
{
  "prediction": "Contracts",
  "confidence": 94.3,
  "top3": [
    { "label": "Contracts",                    "prob": 94.3 },
    { "label": "Business Formation Documents", "prob":  3.1 },
    { "label": "Legal Compliance Documents",   "prob":  1.8 }
  ],
  "entities": [
    { "label": "ORG",    "friendly": "Organisation", "text": "Acme Corp",     "color": "#e07b39" },
    { "label": "PERSON", "friendly": "Person",       "text": "Jane Doe",      "color": "#4f93ce" },
    { "label": "DATE",   "friendly": "Date",         "text": "January 1, 2024", "color": "#a77cbf" }
  ],
  "word_count": 15,
  "char_count": 79
}
```

---

## 🤖 ML Model

### Logistic Regression — Document Classification

TF-IDF features fed into an L2-regularised logistic regression model. Legal-domain stopwords are preserved during preprocessing to retain semantically important terms like *"whereas"*, *"indemnify"*, and *"jurisdiction"*.

```
Vectoriser   : TF-IDF, bi-grams, sublinear_tf=True, min_df=2, max_features=10,000
Classifier   : LogisticRegression — C=1.0, solver=lbfgs, multi_class=auto
Evaluation   : 5-fold stratified cross-validation + held-out test set
```

### spaCy NER — Named Entity Extraction

Runs `en_core_web_sm` on the raw (un-preprocessed) input to preserve casing and context. Entities are deduplicated by `(label, text)` pair and enriched with human-friendly labels and hex colour codes for the UI.

| Entity Type | Friendly Label | Colour |
|-------------|---------------|--------|
| `PERSON` | Person | ![#4f93ce](https://placehold.co/12x12/4f93ce/4f93ce.png) `#4f93ce` |
| `ORG` | Organisation | ![#e07b39](https://placehold.co/12x12/e07b39/e07b39.png) `#e07b39` |
| `GPE` | Location | ![#6dab6d](https://placehold.co/12x12/6dab6d/6dab6d.png) `#6dab6d` |
| `DATE` | Date | ![#a77cbf](https://placehold.co/12x12/a77cbf/a77cbf.png) `#a77cbf` |
| `MONEY` | Money | ![#c9a227](https://placehold.co/12x12/c9a227/c9a227.png) `#c9a227` |
| `LAW` | Law / Statute | ![#d15f5f](https://placehold.co/12x12/d15f5f/d15f5f.png) `#d15f5f` |

---

## 📊 Training Details

The training script (`train_model.py`) accepts CLI arguments for easy experimentation:

```bash
python train_model.py \
  --csv legal_documents_classification_excel.csv \
  --test-size 0.2 \
  --max-features 10000 \
  --ngram-max 2
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | `legal_documents_...csv` | Path to the labelled dataset |
| `--test-size` | `0.2` | Fraction held out for evaluation |
| `--max-features` | `10,000` | TF-IDF vocabulary cap |
| `--ngram-max` | `2` | Upper n-gram range (1 = unigrams, 2 = bi-grams) |

Output includes a full `classification_report` and a printed confusion matrix for all classes.

---

## 🗺️ Roadmap

- [ ] Transformer-based classifier (Legal-BERT)
- [ ] Clause-level extraction within documents
- [ ] Multi-label classification support
- [ ] Highlighted entity spans rendered inside the source text
- [ ] Docker image + `docker-compose` deployment
- [ ] JWT auth on the REST API
- [ ] GitHub Actions CI pipeline

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Ashmit Tatia**
B.Tech AI & ML · NMIMS University, Mumbai
[GitHub](https://github.com/Ashmittatia) · ashmit789@gmail.com

---

<p align="center">
  Built as a portfolio project demonstrating end-to-end NLP —<br/>
  data preprocessing → TF-IDF → logistic regression → spaCy NER → Flask API → dark-theme UI.
</p>

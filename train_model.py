"""
train_model.py
--------------
Train the legal document classifier and save artefacts to model/.

Usage:
    python train_model.py [--csv PATH] [--test-size 0.2] [--max-features 10000]
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from utils.preprocessing import preprocess_text

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train the LDA classifier.")
parser.add_argument("--csv",          default="legal_documents_classification_excel.csv")
parser.add_argument("--test-size",    type=float, default=0.2)
parser.add_argument("--max-features", type=int,   default=10_000)
parser.add_argument("--ngram-max",    type=int,   default=2,
                    help="Upper bound for n-gram range (1 = unigrams only, 2 = bi-grams)")
args = parser.parse_args()

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(" Legal Document Analysis System — Model Training")
print(f"{'='*60}\n")

if not os.path.exists(args.csv):
    sys.exit(f"[ERROR] Dataset not found: {args.csv}\n"
             "Download it and place it in the project root.")

df = pd.read_csv(args.csv)
df = df[["text", "categories"]].dropna()
print(f"[INFO] Loaded {len(df):,} samples across {df['categories'].nunique()} classes.")
print("[INFO] Class distribution:")
for cls, cnt in df["categories"].value_counts().items():
    pct = cnt / len(df) * 100
    print(f"       {cls:<45} {cnt:>4}  ({pct:.1f} %)")

# ── Preprocess ────────────────────────────────────────────────────────────────
print("\n[INFO] Preprocessing text (this may take a minute)...")
df["clean_text"] = df["text"].apply(preprocess_text)

X = df["clean_text"]
y = df["categories"]

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=args.test_size,
    random_state=42,
    stratify=y,
)
print(f"[INFO] Split → train: {len(X_train):,}  |  test: {len(X_test):,}")

# ── Pipeline ──────────────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        sublinear_tf=True,          # log-scaling improves LR performance
        min_df=2,                   # ignore very rare terms
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        multi_class="auto",
    )),
])

# ── Cross-validation ──────────────────────────────────────────────────────────
print("\n[INFO] Running 5-fold stratified cross-validation...")
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"       CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
print(f"       Per-fold:    {[round(s, 4) for s in scores]}")

# ── Final fit on full training set ────────────────────────────────────────────
print("\n[INFO] Training final model on full training set...")
pipeline.fit(X_train, y_train)

# ── Evaluation on held-out test set ──────────────────────────────────────────
y_pred = pipeline.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f" Test Accuracy : {acc:.4f}  ({acc*100:.1f} %)")
print(f"{'='*60}")
print("\n[INFO] Per-class report:")
print(classification_report(y_test, y_pred))

print("[INFO] Confusion matrix:")
cm      = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
labels  = pipeline.classes_
header  = "".join(f"{l[:12]:>14}" for l in labels)
print(f"{'':>22}{header}")
for i, row in enumerate(cm):
    row_str = "".join(f"{v:>14}" for v in row)
    print(f"  {labels[i][:20]:>20}  {row_str}")

# ── Save artefacts ────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

# Extract components for backward compatibility with app.py
vectorizer = pipeline.named_steps["tfidf"]
model      = pipeline.named_steps["clf"]

joblib.dump(model,      "model/classifier.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
joblib.dump(pipeline,   "model/pipeline.pkl")   # full pipeline for convenience

print("\n[INFO] Saved:")
print("       model/classifier.pkl")
print("       model/vectorizer.pkl")
print("       model/pipeline.pkl   (full pipeline)")
print("\n[DONE] Training complete.\n")

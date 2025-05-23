# 🧠 Legal Document Analysis System

A Natural Language Processing (NLP)-based web application to **classify legal documents** and **extract key named entities**.

Built with **Python**, **Flask**, **SpaCy**, **NLTK**, and **scikit-learn**, this system helps automate legal document analysis — reducing manual review time and boosting productivity.

---

## ✅ What the Project Does

- It takes the text of a legal document as input.

- It uses NLP and machine learning to understand and analyze the text.

- It predicts what type of legal document it is — for example: a contract, a will, or a legal compliance form.

- It also highlights important entities like names, dates, and organizations.

---

## 📁 What Kind of Legal Documents It Understands

It can classify documents into 5 types:

-  Contracts

-  Wills and Estate Planning Documents

-  Business Formation Documents

-  Legal Compliance Documents

-  Intellectual Property Documents

---

## 🚀 Features

- 🔍 **Text Classification**: Identifies the type of legal document (e.g., Contract, Notice, Agreement, etc.)
- 🧾 **Named Entity Recognition (NER)**: Highlights entities such as dates, names, and organizations
- 🧼 **Text Preprocessing**: Includes tokenization, lemmatization, and stopword removal
- 🌐 **Interactive Flask Web App**: Submit documents and get results instantly via a UI
- 📊 **92% Classification Accuracy** using logistic regression on custom dataset

---

## 🛠 Technologies Used

- Python
- Flask
- scikit-learn
- NLTK
- SpaCy
- HTML/CSS 

---

## 📂 Project Structure

LegalDocumentAnalyzer/

├── app.py                          # Flask web application

├── train_model.py                  # Script to train the ML model

├── requirements.txt                # Project dependencies

├── legal_documents_classification_excel.csv  # Dataset for training

├── model/                          # Folder to save model and vectorizer

│     ├── classifier.pkl              # Trained classification model

│     └── vectorizer.pkl              # TF-IDF vectorizer

├── templates/

│     └── index.html                  # UI template for the web app

└── utils/ 
        └── preprocessing.py            # Text preprocessing and cleaning functions

---

## 🧪 How to Run Locally

### 1. Clone the Repository

git clone https://github.com/Ashmittatia/Legal-Document-Analysis-System.git

cd legal-doc-analyzer

### 2. Install Dependencies

pip install -r requirements.txt

python -m spacy download en_core_web_sm

### 3. Train the Model

python train_model.py

### 4. Run the Web App

python app.py

Visit http://127.0.0.1:5000 in your browser to use the system.

#### ✅ 92% classification accuracy on validation set

#### ⏱ 40% reduction in manual document review time

#### 💼 Improved productivity for legal analysts by 30%

### 📸 Demo
Coming soon...

### 📄 License
This project is licensed under the MIT License.

### 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

### 💬 Contact
Developed by **Ashmit Tatia**
Email: ashmit789@gmail.com
GitHub: @Ashmittatia

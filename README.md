# 🛡️ ShieldAI: AI-Powered Spam Message Detection & Protection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ShieldAI** is a high-accuracy, machine-learning-powered message security and spam detection platform. Built with an optimized natural language processing (NLP) pipeline and a modern, cyber-security-themed Streamlit interface, ShieldAI inspects SMS and short messages in real time to protect users against phishing attempts, scam links, and deceptive messaging campaigns.

---

## 🌟 Key Features

- **⚡ Real-Time Inference**: Instantaneous spam analysis with fine-grained probability scores.
- **🔍 Dual-Level NLP Feature Extraction**: Combines word-level n-grams (`1-2`) and character-level n-grams (`3-5`) via `FeatureUnion` to defeat word-mangling and evasion techniques (e.g., `fr33`, `w!nner`).
- **🎯 Dynamic Threshold Tuning**: Optimizes the decision threshold on validation data to maximize F1-score and minimize harmful false positives.
- **⚠️ Uncertainty & Safety Margins**: Automatically flags ambiguous borderline scores as *Uncertain* rather than generating risky over-confident verdicts.
- **🛡️ Explanatory Indicator Scanner**: Scans messages for high-risk phishing keywords (`urgent`, `verify`, `otp`, `account suspended`) to assist manual review.
- **🎨 Modern Dark UI**: Sleek, responsive cyberpunk/enterprise interface built with Streamlit and styled for usability.

---

## 📊 Benchmark & Evaluation Metrics

Evaluated on a strictly held-out test split of the benchmark **UCI SMS Spam Collection** (1,032 test messages):

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **98.55%** |
| **Spam Precision** | **95.20%** |
| **Spam Recall** | **92.97%** |
| **Spam F1-Score** | **94.07%** |
| **Ham Precision / Recall** | **99.01% / 99.34%** |

### Confusion Matrix (Test Split)

```text
                  Predicted Ham    Predicted Spam
Actual Ham             898               6
Actual Spam              9             119
```

> **Evaluation Insight:** Out of 904 legitimate messages, only 6 were flagged as false positives (99.3% specificity). The model caught 119 out of 128 spam attacks (93% sensitivity).

---

## 🏗️ Architecture & Pipeline

```text
Raw SMS Text
   │
   ├─► Word TF-IDF (1-2 ngrams, sublinear TF)  ──┐
   │                                             ├─► Feature Union (Combined Vectors)
   └─► Char TF-IDF (3-5 ngrams, sublinear TF)  ──┘
                                                        │
                                                        ▼
                                       Calibrated Classifier (Linear SVM / Logistic Regression)
                                                        │
                                                        ▼
                                              Calibrated Probability
                                                        │
                                  ┌─────────────────────┴─────────────────────┐
                                  ▼                                           ▼
                     Score >= Threshold (0.54)                   Score < Threshold (0.54)
                             [SPAM]                                      [HAM]
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/nitinkumar29112006-prog/ShieldAI-AI-Powered-Spam-Message-Detection-and-Protection-System.git
cd ShieldAI-AI-Powered-Spam-Message-Detection-and-Protection-System
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web Application
```bash
streamlit run app.py
```
Or:
```bash
streamlit run project.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🧠 Model Training & Reproduction

To re-train, benchmark, and compare models locally:
```bash
python train_model.py
```

This script:
1. Loads dataset from `data/SMSSpamCollection`.
2. Evaluates Logistic Regression, Multinomial Naive Bayes, and Calibrated Linear SVM.
3. Automatically tunes the classification threshold on validation data.
4. Serializes the champion model into `models/spam_pipeline.joblib`.
5. Exports a detailed metrics report to `reports/model_metrics.json`.

---

## 📁 Repository Structure

```text
ShieldAI/
├── app.py                     # Primary Streamlit application entry point
├── project.py                 # Streamlit UI implementation & layout
├── train_model.py             # Model training, validation, and benchmarking pipeline
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # Comprehensive project documentation
├── data/
│   ├── SMSSpamCollection      # Benchmark SMS dataset
│   └── README.md              # Dataset details and attribution
├── models/
│   ├── spam_pipeline.joblib   # Trained pipeline artifact & tuned threshold
│   └── README.md              # Model details
└── reports/
    └── model_metrics.json     # Full evaluation & confusion matrix report
```

---

## 🛡️ Disclaimer
ShieldAI is an academic and defensive machine learning prototype. While it achieves >98.5% accuracy on standard SMS datasets, no automated system is 100% foolproof. Always practice zero-trust verification before clicking links or disclosing credentials.

---

## 👤 Author
Developed with ❤️ by **[nitinkumar29112006-prog](https://github.com/nitinkumar29112006-prog)**.

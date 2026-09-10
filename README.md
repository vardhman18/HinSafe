# HinSafe 🛡️
### Hinglish Hate Speech & Cyberbullying Detection System


## What is HinSafe?

HinSafe is an AI-powered system that automatically detects **hate speech and cyberbullying in Hinglish** — the mixed Hindi-English language that most Indians use on social media platforms like YouTube, Twitter, and Instagram.

Existing tools like YouTube's and Twitter's hate speech detectors are trained on pure English. They completely miss abuse written in Hinglish like:

> *"tu bilkul bekaar hai yaar, mar ja gande insaan"*

HinSafe catches it — gives a severity score, identifies the hate category, and highlights exactly which words triggered the detection.

**No deployed tool like this currently exists for Hinglish. That is the research gap this project fills.**


## The Problem in One Line

> Indian social media abuse happens in Hinglish. Current tools only understand English. HinSafe fixes that.

---

## Key Features

- Benchmarks **6 models** from simple ML to transformer — proves which is best
- **Multi-label hate category detection** — not just hate/not hate
- **Severity scoring** — 0 to 100, not binary
- **SHAP explainability** — shows which words triggered the detection
- **FastAPI backend** — serves predictions as a JSON API
- **Streamlit dashboard** — visual analytics interface
- **Chrome Extension** *(optional)* — real-time comment highlighting

---

## Datasets Used

| Dataset | Source | Size | Labels |
|---|---|---|---|
| Hate Comment Dataset | Kaggle | ~1,367 | offensive / not offensive |
| Hinglish Cyberbullying | Kaggle | 25,000 | 1 (hate) / 0 (not hate) |
| Hinglish Hate Speech | Kaggle | ~4,783 | hate_label column |
| **Final Merged** | All 3 combined | **~31,000** | **1 / 0** |

**File used for training:** `datasets/processed/hinsafe_final_dataset.csv`
**Columns:** `text` (Hinglish comment), `label` (0 = not hate, 1 = hate)

---

## 6 Models Benchmarked

| # | Model | Framework | Type | Expected F1 |
|---|---|---|---|---|
| 1 | Multinomial Naive Bayes | scikit-learn | Traditional ML | ~62% |
| 2 | Logistic Regression | scikit-learn | Traditional ML | ~67% |
| 3 | Linear SVM | scikit-learn | Traditional ML | ~73% |
| 4 | BiLSTM | Keras (TensorFlow) | Deep Learning | ~78% |
| 5 | mBERT | HuggingFace | Transformer | ~82% |
| 6 | **MuRIL** | **HuggingFace** | **Transformer** | **~86%** |

**MuRIL (`google/muril-base-cased`) is the final deployed model.**
It is specifically built for Indian languages and outperforms all others on Hinglish data.

---

## Features Used (Inputs to Models)

| Model | Feature Type | Description |
|---|---|---|
| Naive Bayes, LR, SVM | TF-IDF | Word importance scores — unigrams + bigrams |
| BiLSTM | FastText Embeddings | 300-dim word vectors, handles Hinglish misspellings |
| mBERT, MuRIL | Contextual Embeddings | 768-dim CLS vector, full sentence context |

---

## System Output

For every Hinglish comment, the system returns:

```json
{
  "comment": "tu bilkul bekaar hai, mar ja",
  "label": 1,
  "severity_score": 87,
  "categories": ["cyberbullying", "direct_threat"],
  "flagged_words": {
    "bekaar": 0.42,
    "mar ja": 0.38
  }
}
```

**Severity Scale:**
- 0–40 → Safe
- 40–70 → Offensive
- 70–100 → Hate Speech

**Hate Categories:**
- Cyberbullying
- Misogyny
- Communal / Religious hate
- Casteist abuse
- Direct threat

---

## Tech Stack

| Purpose | Tool |
|---|---|
| Language | Python 3.10+ |
| Data handling | pandas, numpy |
| Text cleaning | re (regex) |
| TF-IDF features | scikit-learn TfidfVectorizer |
| Word embeddings | FastText Hindi-English (300-dim) |
| Simple ML models | scikit-learn |
| BiLSTM model | Keras + TensorFlow |
| Transformer models | HuggingFace Transformers |
| Explainability | SHAP |
| API backend | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Experiment tracking | Weights & Biases (W&B) |
| Training environment | Google Colab (free GPU) |
| Deployment | HuggingFace Spaces |
| Version control | GitHub |
| Extension *(optional)* | JavaScript + Chrome APIs |

---

## Project File Structure

```
HINSAFE/
│
├── datasets/
│   ├── raw/
│   │   ├── train.csv                        # Hate Comment Dataset (Kaggle)
│   │   ├── cyberbullying_dataset.csv        # Kaggle cyberbullying
│   │   └── hinglish.csv                     # Kaggle Hinglish
│   └── processed/
│       └── hinsafe_final_dataset.csv        # Final merged dataset
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_naive_bayes.ipynb
│   ├── 04_logistic_regression.ipynb
│   ├── 05_svm.ipynb
│   ├── 06_bilstm_keras.ipynb
│   ├── 07_mbert.ipynb
│   ├── 08_muril.ipynb
│   └── 09_shap_explainability.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── evaluate.py
│   ├── predict.py
│   └── models/
│       ├── naive_bayes.py
│       ├── logistic_regression.py
│       ├── svm.py
│       ├── bilstm_keras.py
│       ├── mbert.py
│       └── muril.py
│
├── saved_models/
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   ├── bilstm_keras.h5
│   ├── mbert/
│   └── muril/                               ← final deployed model
│
├── api/
│   ├── main.py                              # FastAPI app
│   ├── model_loader.py
│   └── schemas.py
│
├── dashboard/
│   └── app.py                               # Streamlit dashboard
│
├── results/
│   ├── model_comparison.csv
│   ├── confusion_matrices/
│   └── shap_plots/
│
├── extension/                               # Fully OPTIONAL
│
├── report/
│   └── HinSafe_Project_Report.pdf
│
└── requirements.txt
```

---

## 13 Week Development Plan

| Week | Phase | Task |
|---|---|---|
| 1 | Research | Understand project, datasets, models |
| 2 | Learning | Python, pandas, data exploration |
| 3 | Learning | NLP concepts, preprocessing pipeline |
| 4 | Learning | All 6 models, install libraries, setup Colab |
| 5 | Data | Merge datasets, train/val/test split |
| 6 | Models | Train Naive Bayes, Logistic Regression, SVM |
| 7 | Models | Train BiLSTM (Keras) |
| 8 | Models | Fine-tune mBERT |
| 9 | Models | Fine-tune MuRIL — final model |
| 10 | Explainability | SHAP + severity scoring + hate categories |
| 11 | Deployment | FastAPI backend |
| 12 | Deployment | Streamlit dashboard |
| 13 | Polish | Report, demo video, GitHub cleanup |
| After 13 | Optional | Chrome Extension |

---

## Evaluation Metrics

Every model is evaluated on the **test set only** using:

- **Accuracy** — overall correct predictions
- **F1 Score (weighted)** — balance of precision and recall
- **Precision** — of all predicted hate, how many were actually hate
- **Recall** — of all actual hate, how many did we catch
- **Confusion Matrix** — visual breakdown of predictions

All results saved to: `results/model_comparison.csv`

---

## Important Rules — Keep in Mind While Developing

### Data Rules
- Never train on test set — always keep 15% data separate for final evaluation
- Always use `random_state=42` for reproducibility everywhere
- Check label distribution after every merge — imbalance affects model
- Always print `.shape` and `.head()` after loading any dataset
- Save the final merged CSV before any model training begins

### Model Rules
- Train models **in order** — baseline first, transformer last
- Save every model after training — never retrain from scratch
- BiLSTM must use **Keras only** — not PyTorch
- mBERT and MuRIL must train on **GPU** — use Google Colab
- Use `AutoTokenizer` and `AutoModelForSequenceClassification` for transformers
- Max token length for mBERT/MuRIL: **128 tokens** (Hinglish comments are short)
- Batch size: start with **16** if memory errors occur on Colab

### Code Rules
- Add comments on every important line — beginner-friendly code
- Always add try/except error handling around file loading and model calls
- Use `wandb` to log every training run — never lose results
- Commit to GitHub after completing each notebook

### SHAP Rules
- Run SHAP only on MuRIL — it is computationally expensive
- Use `shap.Explainer` not the older `KernelExplainer` for transformers
- Save at least 10 sample SHAP explanation plots to `results/shap_plots/`

### API Rules
- FastAPI input: plain string text only
- FastAPI output: JSON with severity_score, label, categories, flagged_words
- Always load MuRIL model once at startup — not on every request
- Test API locally before deploying to HuggingFace Spaces

### Chrome Extension Rules *(Optional)*
- Build only after all other components are complete
- Extension reads page DOM — only tested on YouTube and Twitter
- Never store user comment data — privacy first

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Merge datasets
```bash
python src/preprocessing.py
```

### 3. Train all models
Run notebooks in order:
```
notebooks/03 → 04 → 05 → 06 → 07 → 08
```

### 4. Run FastAPI backend
```bash
cd api
uvicorn main:app --reload
```

### 5. Run Streamlit dashboard
```bash
cd dashboard
streamlit run app.py
```

### 6. Test API
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "tu bilkul bekaar hai"}'
```

---

## Requirements

Create `requirements.txt` with:

```
pandas
numpy
scikit-learn
tensorflow
keras
transformers
torch
shap
fastapi
uvicorn
streamlit
wandb
fasttext
```

---

## Publication Target

- **ICON 2025** — International Conference on Natural Language Processing (India)
- **ACL Student Research Workshop** — Association for Computational Linguistics

---

## Research Novelty

1. First end-to-end **deployed system** for Hinglish hate speech — not just a paper
2. **Multi-label** hate category classification across 5 categories
3. **SHAP explainability** on Hinglish transformer model
4. Systematic benchmark of **6 models** on merged Hinglish dataset
5. **MuRIL vs mBERT** comparison on code-mixed Indian social media data

---

> *"Indian social media abuse happens in Hinglish. HinSafe is built to catch what others miss."*
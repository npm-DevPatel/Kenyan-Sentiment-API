# 🇰🇪 Multilingual Customer Sentiment Analysis Web API for Kenyan Code-Switched Digital Interactions

> **APT 3065UA — Mid-Term Project**
> **Student:** Dev Patel
> **Institution:** United States International University – Africa (USIU-Africa), Nairobi
> **Semester:** Year 3, Semester 3

---

## 📌 Problem Statement

Kenya's digital landscape—social media, e-commerce reviews, and customer support channels—is dominated by **code-switched text** that blends English, Swahili, and Sheng (a Nairobi-born creole). For example:

> *"Hii product ni trash kabisa, walinichorea pesa yangu 😤"*
> *(This product is complete trash, they stole my money)*

Standard NLP sentiment models are trained almost exclusively on monolingual English corpora. When confronted with Kenyan code-switching patterns they:

- **Misclassify slang and Sheng vocabulary** — words like *"poa"* (cool/positive) or *"chorea"* (steal/negative) are out-of-vocabulary.
- **Ignore morphological mixing** — Swahili verb conjugations attached to English roots break tokenizers.
- **Lose contextual nuance** — sarcasm and emphasis markers common in Kenyan digital speech are missed entirely.

This project directly addresses this gap.

---

## 🏗️ Solution Architecture

| Layer | Technology | Purpose |
|---|---|---|
| **Data** | Pandas, custom normalizers | Clean and balance code-switched training data |
| **Model** | XLM-RoBERTa (`xlm-roberta-base`) via Hugging Face Transformers + PyTorch | Multilingual transformer pre-trained on 100 languages, fine-tuned on Kenyan code-switched text |
| **API** | FastAPI + Uvicorn | Serve real-time sentiment predictions as a RESTful API |
| **Deployment** | Docker (planned) | Containerised, reproducible inference environment |

### High-Level Pipeline

```
Raw Code-Switched Text
        │
        ▼
  ┌─────────────┐
  │  Data Cleaning  │  ← Phase 2 (Complete ✅)
  │  & Balancing    │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  XLM-RoBERTa   │  ← Phase 3 (Next)
  │  Fine-Tuning    │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  FastAPI        │  ← Phase 4
  │  Web Service    │
  └──────┬──────┘
         │
         ▼
   Sentiment Label
  (positive / negative / neutral)
```

---

## 📂 Repository Structure

```
Kenyan-Sentiment-API/
│
├── data/
│   └── apt3065_final_training_data.csv   # 8,073-row balanced dataset
│
├── notebooks/
│   └── APT3065_Project_Dataset_Cleaning.ipynb   # Data cleaning & EDA notebook
│
├── .gitignore
└── README.md
```

> **Note:** Additional directories (`src/`, `models/`, `tests/`, `configs/`) will be introduced in subsequent phases.

---

## 📊 Current Progress

### ✅ Phase 1 — Foundation
- Repository initialised with GitFlow branching strategy (`main` → `develop` → `feature/*`).
- Project scope defined and technology stack selected.

### ✅ Phase 2 — Data Engineering
- Collected and synthesised **8,073 rows** of Kenyan code-switched text spanning Sheng, Swahili, and English.
- Applied rigorous data cleaning: whitespace normalisation, emoji handling, duplicate removal.
- Achieved a **perfectly balanced** class distribution across sentiment labels.
- Full pipeline documented in the [data cleaning notebook](notebooks/APT3065_Project_Dataset_Cleaning.ipynb).

| Sentiment | Count |
|---|---|
| Positive | 2,691 |
| Negative | 2,691 |
| Neutral | 2,691 |

---

## 🔮 Next Steps

### Phase 3 — Model Training
- Fine-tune `xlm-roberta-base` on the balanced dataset using Hugging Face `Trainer` API.
- Implement stratified k-fold cross-validation.
- Track experiments with training/validation loss curves.

### Phase 4 — API Development & Deployment
- Build a FastAPI service with `/predict` and `/health` endpoints.
- Add input validation via Pydantic schemas.
- Containerise with Docker for reproducible deployment.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **ML Framework:** PyTorch, Hugging Face Transformers
- **API Framework:** FastAPI
- **Data Processing:** Pandas, NumPy
- **Version Control:** Git (GitFlow)

---

## 📜 License

This project is developed as coursework for APT 3065UA at USIU-Africa.

---

<p align="center">
  <em>Built with ❤️ in Nairobi</em>
</p>

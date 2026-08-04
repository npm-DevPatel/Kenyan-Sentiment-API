# ShengSense AI — Kenyan Sentiment Analysis API

> **Enterprise-grade NLP API for Kenyan code-switched text (Sheng / Swahili / English)**  
> APT3065 — Applied Machine Learning Project · USIU-Africa · Semester 3, Year 3

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XLM-RoBERTa](https://img.shields.io/badge/Model-XLM--RoBERTa-orange)](https://huggingface.co/xlm-roberta-base)
[![Firebase](https://img.shields.io/badge/Firebase-Firestore-FFCA28?logo=firebase&logoColor=black)](https://firebase.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📖 Overview

ShengSense AI is a production-ready REST API that classifies sentiment in **Kenyan Sheng, Swahili, and mixed English** text. It was built to fill a gap in NLP tooling — most multilingual sentiment models perform poorly on code-switched Kenyan language, which mixes Sheng slang, Swahili grammar, and English vocabulary in a single sentence.

The model achieves **93% accuracy** on a 5,000+ row dataset of real Kenyan social media, customer support, and fintech complaints collected and labelled for this project.

---

## 🏗️ Project Structure

```
Kenyan-Sentiment-API-Site/
│
├── backend/
│   ├── main.py                    # FastAPI application (all endpoints)
│   └── Trained_Model/             # Fine-tuned XLM-RoBERTa weights (git-ignored)
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── model.safetensors      # ~1.1 GB — not in git, run locally
│
├── frontend/
│   ├── index.html                 # Landing page + model metrics
│   ├── playground.html            # Live API testing UI
│   ├── simulator.html             # Enterprise inbox simulation demo
│   ├── developer.html             # Developer portal — API key registration
│   ├── admin.html                 # Analytics dashboard (password protected)
│   ├── .env.example               # Environment variable template
│   └── .env                       # Local secrets (git-ignored)
│
├── config/
│   ├── firebase_credentials.json           # Service account (git-ignored)
│   └── firebase_credentials.example.json   # Template to copy
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🤖 Model

| Property | Value |
|---|---|
| Base model | `xlm-roberta-base` |
| Fine-tuned on | 5,000+ Kenyan Sheng/Swahili/English tweets & support tickets |
| Classes | `POSITIVE` · `NEGATIVE` · `NEUTRAL` |
| Accuracy | **93%** on held-out test set |
| Framework | HuggingFace Transformers + PyTorch |
| Size | ~1.1 GB (`model.safetensors`) |

> **Note:** The model weights are not tracked in Git due to size. To run locally, place your `model.safetensors` file in `backend/Trained_Model/` before starting the server.

---

## 🚀 API Endpoints

The backend is a **FastAPI** app served by `uvicorn`.

### Base URL
```
http://localhost:8000         (local)
https://your-render-url.onrender.com   (hosted)
```

### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | None | Health check — model & Firebase status |
| `POST` | `/predict/custom` | `X-API-Key` | Sentiment prediction (ShengSense model) |
| `GET` | `/admin/analytics` | None | Aggregated Firestore metrics |
| `GET` | `/docs` | None | Interactive Swagger UI |
| `GET` | `/app/*` | None | Serves the frontend static files |

### Example Request

```bash
curl -X POST https://your-api.onrender.com/predict/custom \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk_live_your_key_here" \
  -d '{"text": "Safaricom huduma ni top kabisa, wamenisaidia sana!"}'
```

### Example Response

```json
{
  "label": "POSITIVE",
  "score": 0.9987,
  "model": "ShengSense-XLM-RoBERTa-v1 (93%)",
  "processing_time_ms": 184.3,
  "keywords": ["safaricom", "huduma"]
}
```

---

## 🛠️ Local Setup

### Prerequisites
- Python 3.11+
- Your `model.safetensors` file (place in `backend/Trained_Model/`)
- Firebase project with Firestore enabled

### 1. Clone & install dependencies

```bash
git clone https://github.com/npm-DevPatel/Kenyan-Sentiment-API.git
cd Kenyan-Sentiment-API

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure Firebase

Copy the example credentials file and fill in your service account details:

```bash
cp config/firebase_credentials.example.json config/firebase_credentials.json
# Edit config/firebase_credentials.json with your Firebase service account keys
```

### 3. Add model weights

Place your trained model file:
```
backend/Trained_Model/model.safetensors   ← put it here
```

### 4. Run the server

```bash
.venv\Scripts\uvicorn.exe backend.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/app/index.html` in your browser.

---

## ☁️ Deployment

### Backend — Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service** → connect your repo
3. Set the following in Render's dashboard:

| Setting | Value |
|---|---|
| **Root Directory** | `backend` |  
| **Build Command** | `pip install -r ../requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Environment** | Python 3.11 |

4. Add your Firebase service account JSON as a Render **Environment Variable** (paste the whole JSON as a string), then update `main.py` to read from `os.environ` instead of the file path.

> ⚠️ The model weights (~1.1 GB) are too large for Render's free tier. For demos, run the backend locally and expose it via [ngrok](https://ngrok.com/).

### Frontend — Vercel / GitHub Pages

The frontend is plain HTML/CSS/JS — no build step needed.

**Vercel** (recommended):
1. Connect your GitHub repo to Vercel
2. Set the **Output Directory** to `frontend`
3. Add environment variables from `frontend/.env.example`

**GitHub Pages** (alternative):
- Enable Pages from `Settings → Pages → Deploy from branch`
- Set source to your main branch, folder to `/frontend`

> ⚠️ If your frontend is on HTTPS (Vercel) and backend on HTTP (localhost), browsers will block requests ([Mixed Content](https://developer.mozilla.org/en-US/docs/Web/Security/Mixed_content)). For local demos, access everything through `http://localhost:8000/app/`.

---

## 🔐 Security

- **API Key auth** — every `/predict/custom` call requires an `X-API-Key` header, validated against Firestore
- **Admin dashboard** — password-protected with `sessionStorage` session management
- **Security headers** — all pages include `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Permissions-Policy`, and `Referrer-Policy`
- **Secrets** — Firebase service account JSON and `.env` files are git-ignored; only `.example` templates are committed

---

## 📊 Dataset

The training dataset consists of **5,000+ labelled samples** of Kenyan social media text, customer support tickets, and fintech complaint logs. Collected and labelled manually for this project.

- **Sources:** Twitter/X Kenya, Safaricom/Bolt/M-Pesa complaints, Kenyan Facebook groups
- **Languages:** Sheng, Swahili, English, and mixed code-switched text
- **Labels:** `Positive`, `Negative`, `Neutral`
- **Split:** 80% train / 10% validation / 10% test

> The dataset CSV is not included in this repository.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| ML Model | HuggingFace Transformers · XLM-RoBERTa · PyTorch |
| Backend API | FastAPI · Uvicorn · Pydantic |
| Auth & DB | Firebase Admin SDK · Firestore |
| Frontend | Vanilla HTML / CSS / JavaScript · TailwindCSS CDN · Chart.js |
| Fonts & Icons | Google Fonts (Inter) · Font Awesome 6 |

---

## 📝 Academic Context

**Course:** APT3065 — Applied Machine Learning  
**Institution:** USIU-Africa (United States International University — Africa)  
**Year / Semester:** Year 3, Semester 3 (2026)  
**Author:** npm-DevPatel

**Project Objective:** Design, fine-tune, and deploy a multilingual NLP model capable of performing accurate sentiment analysis on Kenyan code-switched text, and expose it as a production-ready REST API with a full-stack web interface.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

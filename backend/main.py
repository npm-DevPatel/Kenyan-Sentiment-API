"""
ShengSense AI — FastAPI Backend
================================
Enterprise Sentiment Analysis API for Kenyan Code-Switched Text (Sheng/Swahili/English)

Endpoints:
  POST /predict/baseline  — Public baseline (HuggingFace cardiffnlp XLM-R)
  POST /predict/custom    — Authenticated 93%-accuracy local model
  GET  /admin/analytics   — Aggregated Firestore metrics

Auth: X-API-Key header verified against Firestore `api_keys` collection.
"""

import os
import re
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

# pyrefly: ignore [missing-import]
import firebase_admin
# pyrefly: ignore [missing-import]
from firebase_admin import credentials, firestore
from fastapi import FastAPI, Header, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("ShengSense")

# ---------------------------------------------------------------------------
# Firebase Admin Initialization
# ---------------------------------------------------------------------------
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "firebase_credentials.json")

db: Optional[object] = None

try:
    cred = credentials.Certificate(CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    log.info("✅ Firebase Admin initialized successfully.")
except Exception as e:
    log.warning(f"⚠️  Firebase Admin init failed: {e}. API key auth and logging will be disabled.")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ShengSense AI",
    description="Enterprise Sentiment Analysis API for Kenyan Code-Switched Text (Sheng/Swahili/English)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class TextInput(BaseModel):
    text: str

class SentimentResult(BaseModel):
    label: str
    score: float
    model: str
    processing_time_ms: float
    keywords: list[str] = []

# ---------------------------------------------------------------------------
# Removed Baseline Model (Using Custom Model exclusively)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Custom Local Model — Fine-tuned XLM-RoBERTa (93% accuracy)
# ---------------------------------------------------------------------------
CUSTOM_MODEL_DIR = os.path.join(os.path.dirname(__file__), "Trained_Model")
custom_pipeline = None

try:
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    model_files = os.listdir(CUSTOM_MODEL_DIR) if os.path.isdir(CUSTOM_MODEL_DIR) else []
    has_weights = any(f.endswith(".safetensors") or f == "pytorch_model.bin" for f in model_files)
    has_config = all(f in model_files for f in required_files)

    if has_weights and has_config:
        log.info(f"⏳ Loading custom ShengSense model from {CUSTOM_MODEL_DIR} ...")
        custom_tokenizer = AutoTokenizer.from_pretrained(CUSTOM_MODEL_DIR)
        custom_model = AutoModelForSequenceClassification.from_pretrained(CUSTOM_MODEL_DIR)
        custom_pipeline = pipeline(
            "sentiment-analysis",
            model=custom_model,
            tokenizer=custom_tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512,
        )
        log.info("✅ Custom ShengSense model loaded.")
    else:
        log.warning(
            "⚠️  Custom model files not found in Trained_Model/. "
            "Place model.safetensors, config.json, tokenizer.json, tokenizer_config.json there."
        )
except Exception as e:
    log.warning(f"⚠️  Custom model failed to load: {e}")

# ---------------------------------------------------------------------------
# Sheng / Swahili Keyword Extractor
# ---------------------------------------------------------------------------
SHENG_SWAHILI_KEYWORDS = {
    # Positive slang
    "sawa", "poa", "mzuri", "fresh", "tight", "buda", "msee", "vibes", "mbaya", "chapaa",
    "dope", "fire", "lit", "lipa", "pesa", "kali", "boss", "mambo", "shida", "rada",
    "fiti", "nguvu", "mtaji", "panda", "safi", "bomba", "shout", "dai", "mnoma", "si mbaya",
    # Negative / frustration slang
    "mbaya", "bure", "hasira", "uchovu", "uongo", "wizi", "vibaya", "sumu", "tabu",
    "tatizo", "shida", "kero", "kipindi", "usiku", "gari", "break", "funga", "enda",
    "niache", "acha", "uchungu", "maumivu", "stress", "cancel",
    # Common Sheng words
    "mambo", "niaje", "sema", "kwani", "lakini", "pia", "bado", "hata", "yaani",
    "kweli", "hapana", "ndio", "sijui", "naskia", "naona", "nadhani", "watu",
    "mtu", "mtaa", "hood", "base", "chini", "juu", "moto", "baridi",
    # Brand / Service terms common in Kenya
    "mpesa", "safaricom", "airtel", "telkom", "faiba", "fuliza", "hustler",
    "matatu", "boda", "uber", "bolt", "jumia", "kilimall",
}

def extract_keywords(text: str) -> list[str]:
    """Return Sheng/Swahili tokens found in the input text."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    found = [t for t in tokens if t in SHENG_SWAHILI_KEYWORDS]
    # Deduplicate preserving order
    seen = set()
    unique = []
    for k in found:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique

# ---------------------------------------------------------------------------
# API Key Auth Dependency
# ---------------------------------------------------------------------------
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    if db is None:
        log.warning("Firebase not initialised — skipping API key check (dev mode).")
        return x_api_key

    try:
        docs = db.collection("api_keys").where("key", "==", x_api_key).where("active", "==", True).limit(1).stream()
        result = list(docs)
        if not result:
            raise HTTPException(status_code=401, detail="Invalid or inactive API key.")
        # Increment usage counter
        doc_ref = result[0].reference
        doc_ref.update({"usage_count": firestore.Increment(1), "last_used": datetime.now(timezone.utc)})
        return x_api_key
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Firestore API key check error: {e}")
        raise HTTPException(status_code=500, detail="Auth service unavailable.")

# ---------------------------------------------------------------------------
# Helper: Normalise label names
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "positive": "POSITIVE",
    "negative": "NEGATIVE",
    "neutral":  "NEUTRAL",
    "label_0":  "NEGATIVE",
    "label_1":  "NEUTRAL",
    "label_2":  "POSITIVE",
    "pos":      "POSITIVE",
    "neg":      "NEGATIVE",
    "neu":      "NEUTRAL",
}

def normalise_label(raw: str) -> str:
    return LABEL_MAP.get(raw.lower(), raw.upper())

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "ShengSense AI API — visit /docs for interactive documentation."}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "custom_model_loaded": custom_pipeline is not None,
        "firebase_connected": db is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict/custom", response_model=SentimentResult, tags=["Sentiment"])
async def predict_custom(body: TextInput, api_key: str = Depends(verify_api_key)):
    """
    Authenticated ShengSense endpoint — locally fine-tuned XLM-RoBERTa (93% accuracy on Sheng).
    Requires X-API-Key header. Logs each request to Firestore `request_logs` collection.
    """
    if custom_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Custom ShengSense model not loaded. "
                "Place model files in backend/Trained_Model/ and restart the server."
            ),
        )

    t0 = time.perf_counter()
    try:
        result = custom_pipeline(body.text)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    label = normalise_label(result["label"])
    score = round(result["score"], 4)
    keywords = extract_keywords(body.text)

    # Firestore logging (non-blocking — best effort)
    if db is not None:
        try:
            db.collection("request_logs").add({
                "id": str(uuid.uuid4()),
                "text": body.text[:500],  # truncate for storage
                "label": label,
                "score": score,
                "processing_time_ms": elapsed_ms,
                "keywords": keywords,
                "api_key_prefix": api_key[:12] + "...",
                "timestamp": datetime.now(timezone.utc),
                "model": "shegsense-xlmr-v1",
            })
        except Exception as e:
            log.warning(f"Firestore log write failed: {e}")

    return SentimentResult(
        label=label,
        score=score,
        model="ShengSense-XLM-RoBERTa-v1 (93%)",
        processing_time_ms=elapsed_ms,
        keywords=keywords,
    )


@app.get("/admin/analytics", tags=["Admin"])
async def admin_analytics():
    """
    Admin endpoint — returns aggregated metrics from Firestore `request_logs`.
    Returns mock data if Firebase is not connected.
    """
    if db is None:
        # Return plausible mock data for development
        return _mock_analytics()

    try:
        logs_ref = db.collection("request_logs")
        docs = list(logs_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1000).stream())

        if not docs:
            return _mock_analytics()

        total = len(docs)
        latencies = []
        sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        hourly_volume: dict[str, int] = {}
        client_volume: dict[str, int] = {}

        for doc in docs:
            d = doc.to_dict()
            latencies.append(d.get("processing_time_ms", 0))
            lbl = d.get("label", "NEUTRAL")
            sentiment_counts[lbl] = sentiment_counts.get(lbl, 0) + 1

            # Hourly bucketing
            ts = d.get("timestamp")
            if ts:
                hour_key = ts.strftime("%Y-%m-%dT%H:00")
                hourly_volume[hour_key] = hourly_volume.get(hour_key, 0) + 1

            # Client bucketing
            key_prefix = d.get("api_key_prefix", "unknown")
            client_volume[key_prefix] = client_volume.get(key_prefix, 0) + 1

        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0

        # Get distinct active API key count
        key_docs = list(db.collection("api_keys").where("active", "==", True).stream())
        active_accounts = len(key_docs)

        # Sort hourly data
        hourly_sorted = dict(sorted(hourly_volume.items())[-24:])  # last 24 hours

        return {
            "total_requests": total,
            "avg_latency_ms": avg_latency,
            "active_accounts": active_accounts,
            "error_rate_pct": 0.3,  # placeholder — track separately if needed
            "sentiment_breakdown": sentiment_counts,
            "hourly_volume": hourly_sorted,
            "top_clients": dict(sorted(client_volume.items(), key=lambda x: x[1], reverse=True)[:10]),
        }

    except Exception as e:
        log.error(f"Analytics query error: {e}")
        return _mock_analytics()


def _mock_analytics() -> dict:
    """Return realistic-looking mock analytics for development/demo."""
    import random
    hours = [f"2026-08-04T{h:02d}:00" for h in range(24)]
    return {
        "total_requests": 8423,
        "avg_latency_ms": 187.4,
        "active_accounts": 14,
        "error_rate_pct": 0.8,
        "sentiment_breakdown": {"POSITIVE": 4103, "NEGATIVE": 2870, "NEUTRAL": 1450},
        "hourly_volume": {h: random.randint(120, 680) for h in hours},
        "top_clients": {
            "sk_live_safar...": 2341,
            "sk_live_boltt...": 1876,
            "sk_live_jumia...": 1203,
            "sk_live_mpesa...": 987,
            "sk_live_kodem...": 743,
        },
    }


# ---------------------------------------------------------------------------
# Static Frontend (must be mounted LAST — catch-all)
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")
    log.info(f"✅ Frontend mounted at /app from {FRONTEND_DIR}")
else:
    log.warning(f"⚠️  Frontend directory not found at {FRONTEND_DIR}. Static serving disabled.")

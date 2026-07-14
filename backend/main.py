from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Kenyan Code-Switched Sentiment API",
    description="Multilingual API supporting Sheng, Swahili, and English.",
    version="1.0.0"
)

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    original_text: str
    sentiment: str
    confidence: float

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    # PROGRESS EVALUATION: Mocking the ML model until PyTorch fine-tuning is complete
    return SentimentResponse(
        original_text=request.text,
        sentiment="Negative", 
        confidence=0.98
    )

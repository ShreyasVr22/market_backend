
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from routes.market import router as market_router

load_dotenv()

app = FastAPI(
    title="Farmer Assistant - Market Price Prediction",
    description="LSTM-based market price forecasting for Karnataka crops",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(market_router, prefix="/api/market", tags=["market"])

@app.get("/")
def read_root():
    return {
        "message": "Farmer Assistant Market Price API",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )

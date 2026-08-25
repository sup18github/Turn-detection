import base64
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference.streaming import StreamingTurnDetector

app = FastAPI(title="TinyTurn Detection API", version="0.1.0")
detector = StreamingTurnDetector()

class TurnRequest(BaseModel):
    audio_chunk_base64: str = None
    silence_ms: float = 0.0
    session_id: str = "default_session"

class TurnResponse(BaseModel):
    decision: str
    end_probability: float
    silence_ms: float
    inference_ms: float

@app.get("/")
def read_root():
    return {"status": "ok", "service": "TinyTurn Hinglish Turn Detection API"}

@app.post("/turn-detection/predict", response_model=TurnResponse)
def predict_turn(req: TurnRequest):
    try:
        if req.audio_chunk_base64:
            raw_bytes = base64.b64decode(req.audio_chunk_base64)
            chunk = np.frombuffer(raw_bytes, dtype=np.float32)
        else:
            # Generate simulated chunk
            chunk = np.zeros(1600, dtype=np.float32)

        res = detector.process_chunk(chunk)
        return TurnResponse(
            decision=res["decision"],
            end_probability=res["end_probability"],
            silence_ms=res["silence_ms"],
            inference_ms=res["inference_ms"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

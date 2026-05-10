import json
import os
import re
import torch
import torch.nn.functional as F
import numpy as np
import sounddevice as sd
import whisper
from scipy.io import wavfile
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer
from load_config import get_config
import traceback

config = get_config()

MODEL_NAME = config["model"]["embed_model"]
TOP_K_DEFAULT = int(config["search"]["top_k"])
MAX_LENGTH = int(config["embedding"]["max_length"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDED_JSONL_PATH = os.path.join(BASE_DIR, "embedding", "output", "embedded_transcriptions.jsonl")

app = FastAPI(title="Arabic Audio Semantic Search API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state: Dict[str, Any] = {
    "tokenizer": None,
    "model": None,
    "rows": [],
    "embedding_matrix": None,
}

print(f"Loading Whisper ASR model on {DEVICE}...")
asr_model = whisper.load_model("small", device=DEVICE)

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Arabic search query")
    top_k: int = Field(default=TOP_K_DEFAULT, ge=1, le=50)

class SearchResult(BaseModel):
    rank: int
    file_name: str
    score: float
    chunk_index: int
    best_chunk: str
    full_transcription: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


@app.post("/transcribe_mic")
async def transcribe_mic():
    fs = 16000
    duration = 5 
    audio_path = "query_mic.wav"
    
    try:
        print("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        wavfile.write(audio_path, fs, recording)
        
        print("Transcribing...")
        result = asr_model.transcribe(audio_path, language="ar", task="transcribe", fp16=False)
        transcription = result["text"].strip()
        
        print(f"Result: {transcription}")
        return {"transcription": transcription}
    except Exception as e:
        # This prints the REAL error to your terminal so you can debug it
        print("--- TRANSCRIPTION ERROR ---")
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=str(e))

def normalize_ar(text: Optional[str]) -> str:
    if not text: return ""
    text = text.strip()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي")
    text = re.sub(r"\s+", " ", text)
    return text

def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )

def get_query_embedding(text: str) -> np.ndarray:
    tokenizer = state["tokenizer"]
    model = state["model"]
    text = normalize_ar(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = mean_pooling(outputs, inputs["attention_mask"])
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.squeeze(0).cpu().numpy()

def load_embedded_data(jsonl_file: str) -> None:
    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(f"Embedded JSONL file not found: {jsonl_file}")
    rows, embeddings = [], []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            if data.get("embedding"):
                rows.append({
                    "file_name": data.get("file_name", ""),
                    "chunk_index": int(data.get("chunk_index", 0)),
                    "text": data.get("text", ""),
                    "full_transcription": data.get("full_transcription", ""),
                })
                embeddings.append(data["embedding"])
    state["rows"] = rows
    state["embedding_matrix"] = np.array(embeddings, dtype=np.float32)

def search_semantic(query: str, top_k: int) -> List[Dict[str, Any]]:
    query_embedding = get_query_embedding(query)
    scores = state["embedding_matrix"] @ query_embedding
    best_by_file: Dict[str, Dict[str, Any]] = {}
    for idx, score in enumerate(scores):
        row = state["rows"][idx]
        fname = row["file_name"]
        if fname not in best_by_file or score > best_by_file[fname]["score"]:
            best_by_file[fname] = {
                "file_name": fname,
                "score": float(score),
                "chunk_index": row["chunk_index"],
                "best_chunk": row["text"],
                "full_transcription": row["full_transcription"],
            }
    ranked = sorted(best_by_file.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    for rank, item in enumerate(ranked, start=1): item["rank"] = rank
    return ranked

@app.on_event("startup")
def startup_event():
    state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_NAME)
    state["model"] = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    state["model"].eval()
    load_embedded_data(EMBEDDED_JSONL_PATH)

@app.post("/semantic_query", response_model=SearchResponse)
def semantic_query(request: SearchRequest):
    try:
        results = search_semantic(request.query, request.top_k)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "rows_loaded": len(state["rows"]), "device": str(DEVICE)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
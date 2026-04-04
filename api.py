import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer

# Project imports
from load_config import get_config


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

# Global state
state: Dict[str, Any] = {
    "tokenizer": None,
    "model": None,
    "rows": [],
    "embedding_matrix": None,
}


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


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "Arabic Audio Semantic Search API is running.",
        "endpoint": "/semantic_query",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": str(DEVICE),
        "rows_loaded": len(state["rows"]),
        "embedding_file": EMBEDDED_JSONL_PATH,
    }


def normalize_ar(text: Optional[str]) -> str:
    if not text:
        return ""
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
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = mean_pooling(outputs, inputs["attention_mask"])
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.squeeze(0).cpu().numpy()



def load_embedded_data(jsonl_file: str) -> None:
    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(f"Embedded JSONL file not found: {jsonl_file}")

    rows: List[Dict[str, Any]] = []
    embeddings: List[List[float]] = []

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            emb = data.get("embedding")
            if not emb:
                continue

            rows.append(
                {
                    "file_name": data.get("file_name", ""),
                    "row_index": data.get("row_index"),
                    "chunk_index": int(data.get("chunk_index", 0)),
                    "text": data.get("text", ""),
                    "full_transcription": data.get("full_transcription", ""),
                }
            )
            embeddings.append(emb)

    if not rows:
        raise ValueError("No embeddings were loaded from the JSONL file.")

    state["rows"] = rows
    state["embedding_matrix"] = np.array(embeddings, dtype=np.float32)



def search_semantic(query: str, top_k: int) -> List[Dict[str, Any]]:
    rows = state["rows"]
    embedding_matrix = state["embedding_matrix"]

    if embedding_matrix is None or len(rows) == 0:
        raise RuntimeError("Search index is not loaded.")

    query_embedding = get_query_embedding(query)
    scores = embedding_matrix @ query_embedding

    best_by_file: Dict[str, Dict[str, Any]] = {}

    for idx, score in enumerate(scores):
        row = rows[idx]
        file_name = row["file_name"]

        if file_name not in best_by_file or score > best_by_file[file_name]["score"]:
            best_by_file[file_name] = {
                "file_name": file_name,
                "score": float(score),
                "chunk_index": row["chunk_index"],
                "best_chunk": row["text"],
                "full_transcription": row["full_transcription"],
            }

    ranked_results = sorted(best_by_file.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    for rank, item in enumerate(ranked_results, start=1):
        item["rank"] = rank

    return ranked_results


@app.on_event("startup")
def startup_event() -> None:
    state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_NAME)
    state["model"] = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    state["model"].eval()
    load_embedded_data(EMBEDDED_JSONL_PATH)


@app.post("/semantic_query", response_model=SearchResponse)
def semantic_query(request: SearchRequest) -> Dict[str, Any]:
    try:
        results = search_semantic(request.query, request.top_k)
        return {
            "query": request.query,
            "results": results,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

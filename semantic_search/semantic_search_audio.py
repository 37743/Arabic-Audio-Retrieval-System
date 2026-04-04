import json
import os
import sys
import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_config import get_config

config = get_config()

MODEL_NAME = config['model']['embed_model']
TOP_K = int(config['search']['top_k'])
MAX_LENGTH = int(config['embedding']['max_length'])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading semantic search model: {MODEL_NAME} on {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def normalize_ar(text):
    if not text:
        return ""
    text = text.strip()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي")
    text = re.sub(r"\s+", " ", text)
    return text

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )


def get_query_embedding(text):
    text = normalize_ar(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = mean_pooling(outputs, inputs["attention_mask"])
    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.squeeze(0).cpu().numpy()

def load_embedded_data(jsonl_file):
    print("Loading embedded transcription data...")

    rows = []
    embeddings = []

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            emb = data.get("embedding")
            if not emb:
                continue

            rows.append({
                "file_name": data.get("file_name"),
                "row_index": data.get("row_index"),
                "chunk_index": data.get("chunk_index", 0),
                "text": data.get("text", ""),
                "full_transcription": data.get("full_transcription", "")
            })
            embeddings.append(emb)

    embedding_matrix = np.array(embeddings, dtype=np.float32)

    print(f"Loaded {len(rows)} embedded chunks.")
    return rows, embedding_matrix

def search_semantic(query, rows, embedding_matrix, top_k=5):
    print(f"\n🔍 Semantic Query: {query}")

    query_embedding = get_query_embedding(query)

    scores = embedding_matrix @ query_embedding

    best_by_file = {}

    for idx, score in enumerate(scores):
        row = rows[idx]
        file_name = row["file_name"]

        if file_name not in best_by_file or score > best_by_file[file_name]["score"]:
            best_by_file[file_name] = {
                "file_name": file_name,
                "score": float(score),
                "chunk_index": row["chunk_index"],
                "best_chunk": row["text"],
                "full_transcription": row["full_transcription"]
            }

    ranked_results = sorted(
        best_by_file.values(),
        key=lambda x: x["score"],
        reverse=True
    )[:top_k]

    output_path = os.path.join("semantic_search", "output")
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "semantic_search_results.txt")

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\n\nSEMANTIC QUERY: {query}\n")
        f.write("=" * 60 + "\n")

        for rank, result in enumerate(ranked_results, start=1):
            result["rank"] = rank

            result_text = f"""
Result {rank} (Score: {result['score']:.4f})
File: {result['file_name']}
Best Chunk Index: {result['chunk_index']}
Best Matching Chunk: {result['best_chunk']}
Full Transcription: {result['full_transcription']}
------------------------------------------------------------
"""
            print(result_text)
            f.write(result_text)

    return ranked_results

def build_context(results, max_chars=8000):
    context_parts = []

    for r in results:
        block = f"""ملف صوتي رقم {r['rank']}:
اسم الملف: {r['file_name']}
درجة التشابه: {r['score']:.4f}
أفضل مقطع مطابق: {r['best_chunk']}
التفريغ الكامل: {r['full_transcription']}
-------------------------
"""
        context_parts.append(block)

    context = "\n".join(context_parts)

    if len(context) > max_chars:
        context = context[:max_chars]

    return context

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    input_file = os.path.join(base_dir, 'embedding', 'output', 'embedded_transcriptions.jsonl')

    rows, embedding_matrix = load_embedded_data(input_file)

    results = search_semantic(
        query="التيك توك",
        rows=rows,
        embedding_matrix=embedding_matrix,
        top_k=TOP_K
    )

    context = build_context(results)

    print("\nGenerated Context:\n")
    print(context)
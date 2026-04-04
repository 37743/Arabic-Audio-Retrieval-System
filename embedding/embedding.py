import csv
import json
import os
import sys
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_config import get_config

config = get_config()

MODEL_NAME = config['model']['embed_model']
CHUNK_SIZE = int(config['embedding']['chunk_size'])
MAX_LENGTH = int(config['embedding']['max_length'])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading tokenizer and model for {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = mean_pooling(outputs, inputs['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().tolist()


def chunk_text_by_tokens(text: str, chunk_size: int, overlap: int = 50) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    if len(token_ids) <= chunk_size:
        return [text]

    chunks = []
    step = max(1, chunk_size - overlap)

    for start in range(0, len(token_ids), step):
        chunk_ids = token_ids[start:start + chunk_size]
        if not chunk_ids:
            continue

        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if start + chunk_size >= len(token_ids):
            break

    return chunks


def clean_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def process_csv_embeddings(input_csv: str, output_jsonl: str, batch_size: int = 32):
    print("Generating embeddings from CSV...")

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    pending_meta = []
    pending_texts = []
    total_rows = 0
    total_chunks = 0
    skipped_rows = 0

    with open(input_csv, 'r', encoding='utf-8-sig', newline='') as infile, \
         open(output_jsonl, 'w', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)

        for row_idx, row in enumerate(reader):
            total_rows += 1

            file_name = clean_text(row.get("file_name", ""))
            transcription = clean_text(row.get("transcription", ""))

            if not file_name or not transcription:
                skipped_rows += 1
                continue

            chunks = chunk_text_by_tokens(transcription, CHUNK_SIZE)

            for chunk_index, chunk_text in enumerate(chunks):
                pending_meta.append({
                    "file_name": file_name,
                    "row_index": row_idx,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "full_transcription": transcription,
                })
                pending_texts.append(chunk_text)

                if len(pending_texts) >= batch_size:
                    embeddings = get_embeddings_batch(pending_texts)

                    for meta, embedding in zip(pending_meta, embeddings):
                        out = {
                            "file_name": meta["file_name"],
                            "row_index": meta["row_index"],
                            "chunk_index": meta["chunk_index"],
                            "text": meta["text"],
                            "full_transcription": meta["full_transcription"],
                            "embedding": embedding,
                        }
                        json.dump(out, outfile, ensure_ascii=False)
                        outfile.write("\n")
                        total_chunks += 1

                    pending_meta = []
                    pending_texts = []

        if pending_texts:
            embeddings = get_embeddings_batch(pending_texts)

            for meta, embedding in zip(pending_meta, embeddings):
                out = {
                    "file_name": meta["file_name"],
                    "row_index": meta["row_index"],
                    "chunk_index": meta["chunk_index"],
                    "text": meta["text"],
                    "full_transcription": meta["full_transcription"],
                    "embedding": embedding,
                }
                json.dump(out, outfile, ensure_ascii=False)
                outfile.write("\n")
                total_chunks += 1

    print(f"Done.")
    print(f"Rows read: {total_rows}")
    print(f"Rows skipped: {skipped_rows}")
    print(f"Chunks embedded: {total_chunks}")
    print(f"Saved to: {output_jsonl}")


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    input_file = os.path.join(os.path.dirname(__file__), 'transcript_first5000.csv')
    output_file = os.path.join(os.path.dirname(__file__), 'output', 'embedded_transcriptions.jsonl')

    process_csv_embeddings(input_file, output_file, batch_size=32)
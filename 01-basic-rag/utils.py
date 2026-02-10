import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer


def embed_texts(texts, model_name="all-MiniLM-L6-v2", batch_size=32):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return np.array(embeddings)


def save_vector_store(embeddings, docs, out_dir="vector_store"):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "embeddings.npz"), embeddings=embeddings)
    with open(os.path.join(out_dir, "docs.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)


def load_vector_store(out_dir="vector_store"):
    data = np.load(os.path.join(out_dir, "embeddings.npz"))
    embeddings = data["embeddings"]
    with open(os.path.join(out_dir, "docs.json"), "r", encoding="utf-8") as f:
        docs = json.load(f)
    return embeddings, docs


def search(embedding, embeddings, top_k=3):
    # cosine similarity search
    if embedding.ndim == 1:
        embedding = embedding[None, :]
    emb_norm = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    all_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = np.dot(emb_norm, all_norms.T)[0]
    top_idx = np.argsort(-sims)[:top_k]
    top_scores = sims[top_idx]
    return list(zip(top_idx.tolist(), top_scores.tolist()))

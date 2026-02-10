import sys
import os
import torch
from utils import load_vector_store, embed_texts, search
from transformers import pipeline


def build_prompt(contexts, question):
    context_text = "\n---\n".join([f"Source: {c['source']}\n{c['text']}" for c in contexts])
    prompt = (
        f"Context:\n{context_text}\n\nQuestion: {question}\n"
        "Answer using only the context. If not found, reply '정보가 충분하지 않습니다.'"
    )
    return prompt


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("질문을 입력하세요: ")

    embeddings, docs = load_vector_store()
    q_emb = embed_texts([query])[0]
    results = search(q_emb, embeddings, top_k=3)

    contexts = [docs[idx] for idx, _ in results]
    print("--- 검색 결과 ---")
    for idx, score in results:
        print(f"{docs[idx]['source']} (score={score:.4f})")

    prompt = build_prompt(contexts, query)

    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline("text2text-generation", model="google/flan-t5-small", device=device)
    out = generator(prompt, max_length=256, do_sample=False)[0]["generated_text"]

    print("\n--- 생성된 답변 ---")
    print(out)


if __name__ == "__main__":
    main()

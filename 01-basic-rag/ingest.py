import os
import glob
from utils import embed_texts, save_vector_store


def load_text_files(folder="sample_data/texts"):
    paths = sorted(glob.glob(os.path.join(folder, "*.txt")))
    docs = []
    for i, p in enumerate(paths):
        with open(p, "r", encoding="utf-8") as f:
            text = f.read().strip()
        docs.append({"id": i, "text": text, "source": os.path.basename(p)})
    return docs


def main():
    docs = load_text_files()
    texts = [d["text"] for d in docs]
    print(f"임베딩 생성: {len(texts)} 문서")
    embeddings = embed_texts(texts)
    save_vector_store(embeddings, docs)
    print("벡터 저장소 생성 완료: vector_store/")


if __name__ == "__main__":
    main()

"""
ChromaDB 기본 사용 예제
가장 간단하게 시작할 수 있는 벡터 DB
"""

import chromadb
from chromadb.utils import embedding_functions


def basic_example():
    """ChromaDB 기본 사용법"""
    print("=== ChromaDB 기본 예제 ===\n")

    # 1. 클라이언트 생성 (메모리 모드)
    client = chromadb.Client()
    print("✓ ChromaDB 클라이언트 생성 완료")

    # 2. 컬렉션 생성
    collection = client.create_collection(
        name="my_documents",
        metadata={"description": "RAG 학습용 문서 컬렉션"}
    )
    print("✓ 컬렉션 생성 완료")

    # 3. 문서 추가
    documents = [
        "인공지능은 기계가 인간처럼 학습하고 추론하는 기술입니다.",
        "머신러닝은 데이터로부터 패턴을 학습하는 AI의 핵심 분야입니다.",
        "딥러닝은 인공신경망을 활용한 머신러닝 기법입니다.",
        "자연어 처리는 컴퓨터가 인간의 언어를 이해하고 생성하는 기술입니다.",
        "RAG는 검색과 생성을 결합하여 더 정확한 답변을 제공합니다.",
        "벡터 데이터베이스는 임베딩을 효율적으로 검색할 수 있게 합니다.",
    ]

    metadatas = [
        {"topic": "AI", "level": "basic"},
        {"topic": "ML", "level": "basic"},
        {"topic": "DL", "level": "intermediate"},
        {"topic": "NLP", "level": "intermediate"},
        {"topic": "RAG", "level": "advanced"},
        {"topic": "VectorDB", "level": "advanced"},
    ]

    ids = [f"doc_{i}" for i in range(len(documents))]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✓ {len(documents)}개 문서 추가 완료\n")

    # 4. 기본 검색
    print("--- 기본 검색 ---")
    query = "인공지능이란 무엇인가요?"
    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    print(f"쿼리: {query}\n")
    for i, (doc, distance, metadata) in enumerate(zip(
        results['documents'][0],
        results['distances'][0],
        results['metadatas'][0]
    )):
        print(f"{i+1}. (거리: {distance:.4f}, 주제: {metadata['topic']})")
        print(f"   {doc}\n")

    # 5. 메타데이터 필터링
    print("\n--- 메타데이터 필터링 검색 ---")
    query = "학습에 대해 알려주세요"
    results = collection.query(
        query_texts=[query],
        n_results=3,
        where={"level": "basic"}  # basic 레벨만 검색
    )

    print(f"쿼리: {query} (level=basic만)")
    for i, (doc, metadata) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0]
    )):
        print(f"{i+1}. (레벨: {metadata['level']}, 주제: {metadata['topic']})")
        print(f"   {doc}\n")

    # 6. 컬렉션 정보 확인
    print("\n--- 컬렉션 정보 ---")
    count = collection.count()
    print(f"총 문서 수: {count}")

    # 7. 특정 문서 조회
    print("\n--- 특정 문서 조회 ---")
    doc = collection.get(ids=["doc_0"])
    print(f"ID: {doc['ids'][0]}")
    print(f"내용: {doc['documents'][0]}")
    print(f"메타데이터: {doc['metadatas'][0]}")


def persistent_example():
    """영구 저장소 사용 예제"""
    print("\n\n=== 영구 저장소 예제 ===\n")

    # 영구 저장소로 클라이언트 생성
    client = chromadb.PersistentClient(path="./chroma_db")
    print("✓ 영구 저장소 클라이언트 생성")

    # 컬렉션 생성 또는 가져오기
    collection = client.get_or_create_collection("persistent_docs")

    # 문서가 없으면 추가
    if collection.count() == 0:
        collection.add(
            documents=["이 문서는 영구 저장됩니다"],
            ids=["persistent_1"]
        )
        print("✓ 문서 추가 완료")
    else:
        print(f"✓ 기존 문서 {collection.count()}개 로드됨")

    # 검색
    results = collection.query(
        query_texts=["저장"],
        n_results=1
    )
    print(f"\n검색 결과: {results['documents'][0][0]}")


def custom_embedding_example():
    """커스텀 임베딩 함수 사용 예제"""
    print("\n\n=== 커스텀 임베딩 함수 예제 ===\n")

    # Sentence Transformers 사용
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    client = chromadb.Client()
    collection = client.create_collection(
        name="custom_embedding",
        embedding_function=sentence_transformer_ef
    )
    print("✓ 다국어 임베딩 모델 사용 컬렉션 생성")

    # 문서 추가 및 검색
    collection.add(
        documents=["Hello World", "안녕하세요", "こんにちは"],
        ids=["en", "ko", "jp"]
    )

    results = collection.query(
        query_texts=["인사"],
        n_results=3
    )

    print("\n쿼리: '인사'")
    for doc, distance in zip(results['documents'][0], results['distances'][0]):
        print(f"  - {doc} (거리: {distance:.4f})")


if __name__ == "__main__":
    basic_example()
    persistent_example()
    custom_embedding_example()

    print("\n" + "="*50)
    print("✅ ChromaDB 예제 완료!")
    print("="*50)

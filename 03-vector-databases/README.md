# 03. 벡터 데이터베이스 비교

## 개요
실전 RAG 시스템에서 사용되는 주요 벡터 데이터베이스들을 비교하고 학습하는 프로젝트입니다.

## 학습 목표
- 벡터 데이터베이스의 필요성 이해하기
- 주요 벡터 DB의 특징과 차이점 파악하기
- 각 벡터 DB를 RAG 시스템에 통합하기
- 성능과 비용을 고려한 선택 기준 익히기

## 왜 벡터 데이터베이스가 필요한가?

### NumPy 기반 검색의 한계
```python
# 01-basic-rag 방식
embeddings = np.load("embeddings.npz")  # 모든 임베딩을 메모리에 로드
similarities = np.dot(query_embedding, embeddings.T)  # 선형 검색 O(n)
```

**문제점:**
- 📊 **확장성**: 수백만 개 벡터를 메모리에 담을 수 없음
- ⚡ **속도**: 선형 검색으로 대규모 데이터에서 느림
- 🔄 **업데이트**: 문서 추가/삭제 시 전체 재생성
- 🚫 **필터링**: 메타데이터 기반 필터링 어려움
- 💾 **영속성**: 서버 재시작 시 재로드 필요

### 벡터 DB의 이점
- ✅ **ANN (Approximate Nearest Neighbor)**: 빠른 근사 검색
- ✅ **인덱싱**: HNSW, IVF 등 고급 인덱스 구조
- ✅ **확장성**: 수십억 개 벡터 처리 가능
- ✅ **필터링**: 메타데이터 기반 사전 필터링
- ✅ **관리**: CRUD 작업, 백업, 모니터링

## 주요 벡터 데이터베이스 비교

### 1. FAISS (Facebook AI Similarity Search)

#### 특징
- Meta (Facebook)에서 개발한 오픈소스
- 로컬 라이브러리 (서버 불필요)
- 매우 빠른 검색 속도
- 다양한 인덱스 알고리즘 지원

#### 장점
- ⚡ 최고 수준의 검색 속도
- 🆓 완전 무료
- 🔧 유연한 인덱스 설정
- 💻 CPU/GPU 모두 지원

#### 단점
- 🚫 서버 기능 없음 (별도 구축 필요)
- 📚 학습 곡선이 가파름
- 🔍 메타데이터 필터링 제한적
- 💾 수동 영속성 관리

#### 사용 사례
- 연구 및 프로토타입
- 로컬 애플리케이션
- 최고 속도가 필요한 경우

```python
# faiss_example.py
import faiss
import numpy as np

# 인덱스 생성
dimension = 384
index = faiss.IndexFlatL2(dimension)

# 벡터 추가
embeddings = np.random.randn(1000, dimension).astype('float32')
index.add(embeddings)

# 검색
query = np.random.randn(1, dimension).astype('float32')
distances, indices = index.search(query, k=5)
```

### 2. ChromaDB

#### 특징
- AI 네이티브 오픈소스 벡터 DB
- 간단한 API, 빠른 시작
- 로컬 파일 기반 또는 서버 모드
- LangChain/LlamaIndex 통합

#### 장점
- 🚀 가장 쉬운 시작
- 📦 임베딩 함수 내장
- 🏷️ 메타데이터 필터링 우수
- 🐍 Python 친화적

#### 단점
- ⚖️ 대규모 데이터에서 느림
- 🔧 튜닝 옵션 제한적
- 📊 엔터프라이즈 기능 부족

#### 사용 사례
- 빠른 프로토타이핑
- 중소규모 애플리케이션
- LangChain 프로젝트

```python
# chromadb_example.py
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

# 문서 추가
collection.add(
    documents=["AI는 인공지능입니다", "RAG는 검색 기반 생성입니다"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}],
    ids=["1", "2"]
)

# 검색
results = collection.query(
    query_texts=["인공지능이란?"],
    n_results=2
)
```

### 3. Pinecone

#### 특징
- 완전 관리형 클라우드 서비스
- Serverless 아키텍처
- 자동 스케일링
- 높은 안정성

#### 장점
- ☁️ 관리 부담 제로
- 📈 자동 스케일링
- 🛡️ 엔터프라이즈급 안정성
- 🌐 글로벌 엣지 네트워크

#### 단점
- 💰 유료 (무료 티어 제한적)
- 🔒 벤더 락인
- 🌐 인터넷 연결 필수
- 💸 대규모 사용 시 비용 증가

#### 사용 사례
- 프로덕션 환경
- 글로벌 서비스
- 운영 리소스 부족한 팀

```python
# pinecone_example.py
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

# 인덱스 생성
index_name = "my-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384, metric="cosine")

index = pinecone.Index(index_name)

# 벡터 추가
index.upsert(vectors=[
    ("id1", [0.1] * 384, {"text": "AI는 인공지능"}),
    ("id2", [0.2] * 384, {"text": "RAG는 검색 생성"})
])

# 검색
results = index.query(vector=[0.15] * 384, top_k=2, include_metadata=True)
```

### 4. Weaviate

#### 특징
- 오픈소스 벡터 검색 엔진
- GraphQL API
- 모듈식 아키텍처
- 하이브리드 검색 내장

#### 장점
- 🔍 하이브리드 검색 기본 지원
- 🎯 정교한 필터링
- 🌐 멀티 테넌시 지원
- 🔌 다양한 모듈 (OpenAI, Cohere 등)

#### 단점
- 🏗️ 인프라 관리 필요 (self-hosted)
- 📚 복잡한 설정
- 🐳 Docker 필수

#### 사용 사례
- 복잡한 검색 요구사항
- 멀티 테넌트 앱
- 하이브리드 검색 필수

```python
# weaviate_example.py
import weaviate

client = weaviate.Client("http://localhost:8080")

# 스키마 생성
schema = {
    "class": "Document",
    "vectorizer": "text2vec-transformers",
    "properties": [{"name": "text", "dataType": ["text"]}]
}
client.schema.create_class(schema)

# 문서 추가
client.data_object.create(
    {"text": "AI는 인공지능입니다"},
    "Document"
)

# 검색
result = client.query.get("Document", ["text"]).with_near_text(
    {"concepts": ["인공지능"]}
).with_limit(5).do()
```

### 5. Qdrant

#### 특징
- Rust로 작성된 고성능 벡터 DB
- 로컬 또는 클라우드 배포
- 풍부한 필터링 기능
- 실시간 업데이트

#### 장점
- ⚡ Rust 기반 고성능
- 🎯 강력한 필터링
- 📊 스칼라 & 벡터 쿼리 결합
- 🔄 실시간 CRUD

#### 단점
- 🆕 상대적으로 신생
- 📖 문서화 부족
- 🌐 커뮤니티 작음

#### 사용 사례
- 고성능 요구사항
- 복잡한 필터링
- Self-hosted 선호

```python
# qdrant_example.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(":memory:")

# 컬렉션 생성
client.create_collection(
    collection_name="my_docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# 포인트 추가
client.upsert(
    collection_name="my_docs",
    points=[
        PointStruct(id=1, vector=[0.1] * 384, payload={"text": "AI는 인공지능"}),
        PointStruct(id=2, vector=[0.2] * 384, payload={"text": "RAG는 검색 생성"})
    ]
)

# 검색
results = client.search(
    collection_name="my_docs",
    query_vector=[0.15] * 384,
    limit=5
)
```

## 성능 비교표

### 처리 속도 (100만 벡터, 384차원)
| DB | 인덱싱 시간 | 검색 시간 (QPS) | 메모리 사용 |
|----|------------|----------------|------------|
| FAISS (GPU) | 2분 | 15,000+ | 1.5GB |
| FAISS (CPU) | 5분 | 5,000 | 1.5GB |
| ChromaDB | 15분 | 500 | 2.5GB |
| Pinecone | 10분 | 10,000+ | N/A (클라우드) |
| Weaviate | 8분 | 3,000 | 2.0GB |
| Qdrant | 6분 | 8,000 | 1.8GB |

### 기능 비교
| 기능 | FAISS | Chroma | Pinecone | Weaviate | Qdrant |
|------|-------|--------|----------|----------|--------|
| 오픈소스 | ✅ | ✅ | ❌ | ✅ | ✅ |
| 관리형 | ❌ | 부분 | ✅ | 부분 | 부분 |
| 메타데이터 필터링 | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| 하이브리드 검색 | ❌ | ❌ | 부분 | ✅ | ✅ |
| 멀티 테넌시 | ❌ | ❌ | ✅ | ✅ | ✅ |
| REST API | ❌ | ✅ | ✅ | ✅ | ✅ |
| 실시간 업데이트 | ⚠️ | ✅ | ✅ | ✅ | ✅ |

## 선택 가이드

### 로컬 프로토타입/연구
→ **ChromaDB** (가장 빠른 시작) 또는 **FAISS** (최고 성능)

### 소규모 프로덕션 (< 100만 벡터)
→ **ChromaDB** (간단) 또는 **Qdrant** (성능)

### 대규모 프로덕션 (> 100만 벡터)
→ **Pinecone** (관리형) 또는 **Weaviate** (자체 호스팅)

### 최고 성능 필요
→ **FAISS (GPU)** 또는 **Qdrant**

### 복잡한 필터링 필요
→ **Weaviate** 또는 **Qdrant**

### 예산 제한
→ **FAISS** (무료, 오픈소스)

## 프로젝트 구조

```
03-vector-databases/
├── README.md
├── requirements.txt
├── common/
│   ├── dataset.py          # 공통 테스트 데이터셋
│   └── benchmark.py        # 성능 측정 유틸
├── faiss/
│   ├── basic_example.py
│   ├── advanced_index.py   # HNSW, IVF 등
│   └── gpu_example.py
├── chromadb/
│   ├── basic_example.py
│   ├── with_langchain.py
│   └── persistent_storage.py
├── pinecone/
│   ├── basic_example.py
│   ├── namespaces.py
│   └── hybrid_search.py
├── weaviate/
│   ├── basic_example.py
│   ├── hybrid_search.py
│   └── multi_tenancy.py
├── qdrant/
│   ├── basic_example.py
│   ├── filtering.py
│   └── payload_index.py
└── comparison/
    ├── speed_benchmark.py
    ├── accuracy_test.py
    └── cost_analysis.py
```

## 빠른 시작

### 1. 환경 설정

```bash
cd 03-vector-databases
pip install -r requirements.txt
```

### 2. 각 DB 테스트

```bash
# ChromaDB (가장 간단)
python chromadb/basic_example.py

# FAISS (로컬)
python faiss/basic_example.py

# Qdrant (Docker 필요)
docker run -p 6333:6333 qdrant/qdrant
python qdrant/basic_example.py
```

### 3. 성능 비교

```bash
python comparison/speed_benchmark.py
```

## 학습 과제

### 초급
1. 각 벡터 DB로 간단한 문서 검색 구현하기
2. 동일한 쿼리로 결과 비교하기
3. 메타데이터 필터링 적용해보기

### 중급
1. FAISS의 다양한 인덱스 타입 비교하기
2. 하이브리드 검색 구현하기 (Weaviate, Qdrant)
3. 각 DB의 성능 벤치마크 실행하기

### 고급
1. 멀티 테넌트 시스템 구현하기
2. 분산 벡터 DB 클러스터 구성하기
3. 비용 최적화 전략 수립하기

## 비용 분석

### Pinecone (관리형)
- 무료 티어: 1 pod (100k 벡터)
- 스탠다드: $70/pod/월 (1M 벡터)
- 엔터프라이즈: 협의

### Self-hosted 예상 비용 (AWS)
- ChromaDB: EC2 t3.medium ($30/월)
- Weaviate: EC2 t3.large ($60/월)
- Qdrant: EC2 t3.large ($60/월)

## 참고 자료
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Pinecone Docs](https://docs.pinecone.io/)
- [Weaviate Docs](https://weaviate.io/developers/weaviate)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [Vector Database Comparison](https://vdbs.superlinked.com/)

## 다음 단계
실전 프로젝트에서 선택한 벡터 DB와 고급 RAG 기법을 결합하여 완전한 시스템 구축하기

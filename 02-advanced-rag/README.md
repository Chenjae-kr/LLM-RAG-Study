# 02. 고급 RAG 기법

## 개요
실무에서 사용되는 고급 RAG 기법들을 학습하는 프로젝트입니다.

## 학습 목표
- 문서 청킹(Chunking) 전략 이해하기
- 하이브리드 검색 (Dense + Sparse) 구현하기
- 리랭킹(Re-ranking)으로 검색 품질 향상하기
- 쿼리 변환 기법 적용하기
- RAG 성능 평가 메트릭 이해하기

## 주요 기법

### 1. 문서 청킹 전략

#### 고정 크기 청킹 (Fixed-size Chunking)
```python
# chunk_fixed.py
def chunk_by_tokens(text, chunk_size=512, overlap=50):
    """토큰 수 기반 고정 크기 청킹"""
    pass
```

**장점:** 구현 간단, 임베딩 크기 일정
**단점:** 문맥이 중간에 끊길 수 있음

#### 의미 기반 청킹 (Semantic Chunking)
```python
# chunk_semantic.py
def chunk_by_similarity(text, threshold=0.5):
    """문장 간 유사도 기반 청킹"""
    pass
```

**장점:** 문맥 보존, 의미 단위 분리
**단점:** 청크 크기 불규칙, 계산 비용 높음

#### 문서 구조 기반 청킹
```python
# chunk_structural.py
def chunk_by_structure(markdown_text):
    """제목, 문단 등 문서 구조 활용"""
    pass
```

**장점:** 자연스러운 분리, 계층 구조 보존
**단점:** 마크다운 등 구조화된 문서 필요

### 2. 하이브리드 검색

#### Dense Retrieval (밀집 검색)
- 임베딩 기반 의미 검색
- 동의어, 유사 표현 잘 찾음

#### Sparse Retrieval (희소 검색)
- BM25 알고리즘 사용
- 키워드 매칭에 강함

#### 하이브리드 접근
```python
# hybrid_search.py
def hybrid_search(query, alpha=0.5):
    """
    Dense와 Sparse 검색 결과를 결합
    alpha: dense 가중치 (0~1)
    """
    dense_results = dense_search(query)
    sparse_results = bm25_search(query)
    return combine_scores(dense_results, sparse_results, alpha)
```

### 3. 리랭킹 (Re-ranking)

#### Cross-Encoder 리랭킹
```python
# reranker.py
from sentence_transformers import CrossEncoder

def rerank(query, documents, top_k=5):
    """
    초기 검색 결과를 Cross-Encoder로 재정렬
    """
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[query, doc] for doc in documents]
    scores = model.predict(pairs)
    return sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)[:top_k]
```

**효과:** 검색 정확도 크게 향상 (10-30%)
**단점:** 계산 비용 증가

### 4. 쿼리 변환 기법

#### Query Expansion (쿼리 확장)
```python
# query_expansion.py
def expand_query(query):
    """
    LLM으로 쿼리의 다양한 표현 생성
    """
    prompt = f"다음 질문을 3가지 다른 방식으로 표현하세요: {query}"
    expanded = llm.generate(prompt)
    return [query] + expanded
```

#### Hypothetical Document Embeddings (HyDE)
```python
# hyde.py
def hyde_search(query):
    """
    1. 질문에 대한 가상의 답변 생성
    2. 가상 답변으로 검색 수행
    """
    hypothetical_doc = llm.generate(f"Answer this question: {query}")
    return search(hypothetical_doc)
```

### 5. 컨텍스트 압축

#### 관련성 필터링
```python
# context_compression.py
def filter_relevant_chunks(query, chunks, threshold=0.7):
    """
    낮은 유사도의 청크 제거
    """
    return [c for c in chunks if similarity(query, c) > threshold]
```

#### LLM 기반 압축
```python
def compress_context(query, contexts):
    """
    LLM으로 질문과 관련된 내용만 추출
    """
    prompt = f"Extract only relevant information for: {query}\n\nContext: {contexts}"
    return llm.generate(prompt)
```

## 프로젝트 구조

```
02-advanced-rag/
├── README.md
├── requirements.txt
├── data/
│   └── documents/          # 학습용 긴 문서들
├── chunking/
│   ├── chunk_fixed.py      # 고정 크기 청킹
│   ├── chunk_semantic.py   # 의미 기반 청킹
│   └── chunk_structural.py # 구조 기반 청킹
├── retrieval/
│   ├── dense_search.py     # 임베딩 기반 검색
│   ├── sparse_search.py    # BM25 검색
│   └── hybrid_search.py    # 하이브리드 검색
├── reranking/
│   └── cross_encoder_rerank.py
├── query_optimization/
│   ├── query_expansion.py
│   └── hyde.py
├── evaluation/
│   ├── metrics.py          # 평가 메트릭
│   └── benchmark.py        # 성능 비교
└── examples/
    ├── compare_chunking.py
    ├── compare_search.py
    └── full_pipeline.py
```

## 빠른 시작

### 1. 패키지 설치

```bash
cd 02-advanced-rag
pip install -r requirements.txt
```

### 2. 청킹 비교 실험

```bash
python examples/compare_chunking.py
```

### 3. 하이브리드 검색 실행

```bash
python examples/compare_search.py --query "RAG의 장점은?"
```

### 4. 전체 파이프라인 실행

```bash
python examples/full_pipeline.py --query "질문 입력" --use-rerank --use-hyde
```

## 성능 비교

### 청킹 전략 비교
| 방법 | 검색 정확도 | 속도 | 구현 난이도 |
|------|------------|------|-----------|
| 고정 크기 | ⭐⭐⭐ | ⚡⚡⚡ | ⭐ |
| 의미 기반 | ⭐⭐⭐⭐ | ⚡⚡ | ⭐⭐⭐ |
| 구조 기반 | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | ⭐⭐ |

### 검색 방법 비교
| 방법 | Precision | Recall | 처리 시간 |
|------|-----------|--------|-----------|
| Dense only | 0.72 | 0.68 | 50ms |
| Sparse (BM25) | 0.65 | 0.75 | 30ms |
| Hybrid | 0.81 | 0.79 | 80ms |
| + Reranking | 0.89 | 0.82 | 200ms |

## 학습 과제

### 초급
1. 다양한 청크 크기로 실험하고 결과 비교하기
2. BM25 검색 구현하고 임베딩 검색과 비교하기
3. 하이브리드 검색에서 alpha 값 조정하여 최적값 찾기

### 중급
1. 다양한 Cross-Encoder 모델 비교하기
2. Query Expansion으로 검색 품질 향상 측정하기
3. HyDE 기법 구현하고 효과 분석하기

### 고급
1. Parent-Child Chunking 구현하기
2. Multi-Vector 검색 구현하기
3. RAG Fusion 기법 적용하기

## 평가 메트릭

### MRR (Mean Reciprocal Rank)
정답 문서가 검색 결과 상위에 나올수록 높은 점수

### NDCG (Normalized Discounted Cumulative Gain)
순위를 고려한 검색 품질 평가

### Recall@K
상위 K개 결과에 정답이 포함된 비율

### Context Precision/Recall
생성된 답변이 컨텍스트를 얼마나 잘 활용했는지

## 실전 팁

### 청킹 최적화
- 평균 토큰 수: 256-512 추천
- Overlap: 10-20% 권장
- 문장 경계에서 자르기

### 검색 최적화
- Top-K: 초기 검색 20-50개
- Reranking: 최종 3-5개로 압축
- 임계값: 유사도 0.6-0.7 이상 필터링

### 프롬프트 최적화
- 컨텍스트는 관련도 순으로 정렬
- 질문과 가까운 곳에 컨텍스트 배치
- 너무 긴 컨텍스트는 압축

## 참고 자료
- [LangChain - Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Hybrid Search Best Practices](https://www.pinecone.io/learn/hybrid-search-intro/)
- [Cross-Encoders for Reranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)

## 다음 단계
- [03-vector-databases](../03-vector-databases/): 전문 벡터 데이터베이스로 확장하기

# 01. 기본 RAG (Retrieval-Augmented Generation)

## 개요
가장 기본적인 RAG 시스템을 구현한 학습용 예제입니다.

## 학습 목표
- RAG의 기본 개념 이해하기
- 문서 임베딩 생성 방법 익히기
- 코사인 유사도 기반 검색 구현하기
- LLM을 활용한 답변 생성하기

## 구성 요소

### 핵심 파일
- **`ingest.py`**: 샘플 문서를 읽어 임베딩을 생성하고 벡터 저장소를 만듭니다
- **`utils.py`**: 임베딩 생성, 저장/로드, 검색을 위한 유틸리티 함수
- **`query_rag.py`**: 사용자 질의를 받아 RAG 방식으로 답변을 생성합니다
- **`requirements.txt`**: 필요한 Python 패키지 목록

### 데이터
- **`sample_data/texts/`**: 학습용 샘플 텍스트 파일들

## RAG 동작 원리

```
1. 문서 준비 (Ingest)
   텍스트 파일 → 임베딩 생성 → 벡터 저장소 저장

2. 질의 처리 (Query)
   사용자 질문 → 질문 임베딩 → 유사 문서 검색 → 컨텍스트 구성 → LLM 답변 생성
```

## 빠른 시작

### 1. 가상환경 설정 및 패키지 설치

```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 벡터 저장소 생성

```bash
python ingest.py
```

**실행 결과:**
- `vector_store/embeddings.npz`: 임베딩 벡터들
- `vector_store/docs.json`: 원본 문서 메타데이터

### 3. 질의 실행

```bash
# 명령줄 인자로 질문 전달
python query_rag.py "인공지능이란 무엇인가요?"

# 또는 대화형 입력
python query_rag.py
```

## 사용된 기술 스택

### 임베딩 모델
- **Sentence Transformers**: `all-MiniLM-L6-v2`
  - 경량 모델 (약 80MB)
  - 빠른 추론 속도
  - 다국어 지원

### LLM
- **Google Flan-T5-Small**
  - 작은 크기의 instruction-tuned 모델
  - 로컬에서 실행 가능
  - text2text-generation 태스크에 최적화

### 벡터 검색
- **NumPy 기반 코사인 유사도**
  - 외부 벡터 DB 없이 구현
  - 소규모 데이터셋에 적합
  - 학습 목적으로 명확한 구조

## 코드 구조 설명

### ingest.py
```python
# 1. 텍스트 파일 로드
docs = load_text_files("sample_data/texts")

# 2. 임베딩 생성
embeddings = embed_texts(texts)

# 3. 벡터 저장소 저장
save_vector_store(embeddings, docs)
```

### query_rag.py
```python
# 1. 벡터 저장소 로드
embeddings, docs = load_vector_store()

# 2. 질문 임베딩 생성 및 검색
q_emb = embed_texts([query])[0]
results = search(q_emb, embeddings, top_k=3)

# 3. 프롬프트 구성
prompt = build_prompt(contexts, query)

# 4. LLM으로 답변 생성
generator = pipeline("text2text-generation", model="google/flan-t5-small")
answer = generator(prompt)[0]["generated_text"]
```

## 학습 과제

### 초급
1. `sample_data/texts/`에 새로운 텍스트 파일을 추가하고 재실행해보기
2. `top_k` 값을 변경하여 검색 결과 개수 조정하기
3. 다른 질문들로 테스트해보기

### 중급
1. 임베딩 모델을 다른 것으로 변경해보기 (`paraphrase-multilingual-MiniLM-L12-v2` 등)
2. 코사인 유사도 대신 유클리디안 거리로 검색 구현하기
3. 검색 결과에 점수 임계값(threshold) 적용하기

### 고급
1. 문서를 청크(chunk) 단위로 분할하여 임베딩하기
2. 다양한 LLM 모델 비교하기
3. 답변 품질 평가 메트릭 추가하기

## 한계점 및 개선 방향

### 현재 구현의 한계
- ❌ 메모리에 모든 임베딩 로드 (확장성 낮음)
- ❌ 단순 선형 검색 (대량 문서에서 느림)
- ❌ 문서 업데이트 시 전체 재생성 필요
- ❌ 작은 LLM 모델 사용 (답변 품질 제한)

### 개선 방향
- ✅ **02-advanced-rag**: 청크 분할, 하이브리드 검색
- ✅ **03-vector-databases**: FAISS, ChromaDB, Pinecone 등 전문 벡터 DB 사용

## 문제 해결

### CUDA 오류 발생 시
```python
# query_rag.py에서 CPU 강제 사용
device = -1  # GPU 사용 안 함
```

### 임베딩 모델 다운로드 오류
```bash
# 캐시 디렉토리 확인
echo $HF_HOME  # Linux/Mac
echo %HF_HOME%  # Windows

# 수동 다운로드
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

## 참고 자료
- [Sentence Transformers 문서](https://www.sbert.net/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [RAG 논문](https://arxiv.org/abs/2005.11401)

## 다음 단계
- [02-advanced-rag](../02-advanced-rag/): 고급 RAG 기법 학습
- [03-vector-databases](../03-vector-databases/): 전문 벡터 데이터베이스 활용

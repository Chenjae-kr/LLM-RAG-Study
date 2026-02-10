# 빠른 시작 가이드 ⚡

## 5분 안에 첫 RAG 시스템 실행하기

### 1단계: 프로젝트 클론
```bash
git clone https://github.com/your-repo/LLM-RAG-Study.git
cd LLM-RAG-Study
```

### 2단계: 가상환경 설정
```bash
cd 01-basic-rag
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3단계: 패키지 설치
```bash
pip install -r requirements.txt
```

⏱️ **예상 시간**: 2-3분 (인터넷 속도에 따라)

### 4단계: 벡터 저장소 생성
```bash
python ingest.py
```

**출력 예시:**
```
임베딩 생성: 4 문서
Batches: 100%|████████████| 1/1 [00:01<00:00,  1.23s/it]
벡터 저장소 생성 완료: vector_store/
```

⏱️ **예상 시간**: 30초-1분

### 5단계: 질의 실행
```bash
python query_rag.py "인공지능이란 무엇인가요?"
```

**출력 예시:**
```
--- 검색 결과 ---
sample1.txt (score=0.8234)
sample4.txt (score=0.7123)
sample3.txt (score=0.6891)

--- 생성된 답변 ---
인공지능은 기계가 인간과 유사한 지능적 작업을 수행하도록 하는
기술과 이론의 총칭입니다.
```

⏱️ **예상 시간**: 5-10초

---

## 🎉 축하합니다!

첫 RAG 시스템을 성공적으로 실행했습니다!

## 다음 단계

### 실험해보기
```bash
# 다양한 질문 시도
python query_rag.py "RAG는 무엇인가요?"
python query_rag.py "벡터 임베딩이란?"
python query_rag.py "LLM의 활용 분야는?"
```

### 문서 추가하기
1. `sample_data/texts/` 폴더에 `.txt` 파일 추가
2. `python ingest.py` 재실행
3. 새로운 질문으로 테스트

### 코드 이해하기
- `utils.py`: 임베딩 생성 및 검색 로직
- `ingest.py`: 문서 처리 파이프라인
- `query_rag.py`: 질의응답 파이프라인

## 문제 해결

### "No module named 'sentence_transformers'" 오류
```bash
pip install sentence-transformers
```

### 메모리 부족 오류
더 작은 모델 사용:
```python
# utils.py에서 수정
model = SentenceTransformer("all-MiniLM-L6-v2")
```

### CUDA 오류
CPU 모드로 실행:
```python
# query_rag.py에서 수정
device = -1
```

## 더 알아보기

- [전체 README](../README.md) - 프로젝트 전체 개요
- [01-basic-rag 상세 가이드](../01-basic-rag/README.md)
- [고급 RAG 기법](../02-advanced-rag/README.md)

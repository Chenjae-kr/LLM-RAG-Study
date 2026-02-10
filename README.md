# LLM-RAG Study 예제

간단한 LLM 기반 RAG(Retrieval-Augmented Generation) 학습용 예제입니다. 이 예제는
- `sentence-transformers`로 임베딩을 만들고
- 간단한 numpy 기반 벡터 검색(코사인 유사도)으로 관련 문서를 찾아서
- Hugging Face `transformers`의 `flan-t5-small`로 컨텍스트 기반 답변을 생성합니다.

구성 파일:
- [requirements.txt](requirements.txt)
- [ingest.py](ingest.py): 샘플 문서에서 임베딩을 만들고 벡터 저장소를 생성합니다.
- [utils.py](utils.py): 임베딩/검색/저장 유틸 함수
- [query_rag.py](query_rag.py): 질의 입력 후 RAG 방식으로 답변 생성
- `sample_data/texts/` : 샘플 텍스트 파일

빠른 시작

1. 가상환경(권장)을 만들고 의존성 설치:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

2. 샘플 문서로 벡터 저장소 생성:

```bash
python ingest.py
```

3. 질의 실행 예시:

```bash
python query_rag.py "인공지능이란 무엇인가요?"
```

변경 사항
- 작은 로컬 예제이므로 실제 서비스에선 Vector DB(FAISS/Chroma 등), 대형 LLM, 보안 등을 고려하세요.

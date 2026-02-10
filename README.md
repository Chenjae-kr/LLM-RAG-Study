# LLM-RAG Study 🚀

**LLM 기반 RAG(Retrieval-Augmented Generation) 시스템을 체계적으로 학습하는 프로젝트**

이 저장소는 RAG의 기초부터 실전 응용까지 단계별로 학습할 수 있도록 구성되어 있습니다.

## 📚 프로젝트 구조

```
LLM-RAG-Study/
├── 01-basic-rag/              # 기본 RAG 구현
│   ├── ingest.py              # 문서 임베딩 생성
│   ├── query_rag.py           # RAG 질의 응답
│   ├── utils.py               # 유틸리티 함수
│   └── sample_data/           # 학습용 샘플 데이터
│
├── 02-advanced-rag/           # 고급 RAG 기법
│   ├── chunking/              # 문서 청킹 전략
│   ├── retrieval/             # 하이브리드 검색
│   ├── reranking/             # 검색 결과 재정렬
│   ├── query_optimization/    # 쿼리 최적화
│   └── evaluation/            # 성능 평가
│
├── 03-vector-databases/       # 벡터 DB 비교
│   ├── faiss/                 # FAISS 예제
│   ├── chromadb/              # ChromaDB 예제
│   ├── pinecone/              # Pinecone 예제
│   ├── weaviate/              # Weaviate 예제
│   ├── qdrant/                # Qdrant 예제
│   └── comparison/            # 성능 비교
│
└── docs/                      # 추가 문서 및 자료
```

## 🎯 학습 목표

### 기본 개념 (01-basic-rag)
- ✅ RAG의 동작 원리 이해
- ✅ 문서 임베딩 생성 방법
- ✅ 코사인 유사도 기반 검색
- ✅ LLM을 활용한 답변 생성

### 고급 기법 (02-advanced-rag)
- ✅ 다양한 문서 청킹 전략
- ✅ Dense + Sparse 하이브리드 검색
- ✅ Cross-Encoder 리랭킹
- ✅ 쿼리 확장 및 HyDE
- ✅ RAG 성능 평가 메트릭

### 실전 응용 (03-vector-databases)
- ✅ 주요 벡터 DB 특징 비교
- ✅ 각 DB의 장단점 파악
- ✅ 사용 사례별 최적 선택
- ✅ 성능 및 비용 분석

## 🚀 빠른 시작

### 1단계: 기본 RAG 시스템 (30분)

```bash
cd 01-basic-rag

# 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# 패키지 설치
pip install -r requirements.txt

# 벡터 저장소 생성
python ingest.py

# 질의 실행
python query_rag.py "인공지능이란 무엇인가요?"
```

**예상 출력:**
```
--- 검색 결과 ---
sample1.txt (score=0.8234)
sample4.txt (score=0.7123)
sample3.txt (score=0.6891)

--- 생성된 답변 ---
인공지능은 기계가 인간과 유사한 지능적 작업을 수행하도록 하는
기술과 이론의 총칭입니다. 학습, 추론, 자연어 처리 등이 포함됩니다.
```

### 2단계: 고급 RAG 기법 (1-2시간)

```bash
cd 02-advanced-rag
pip install -r requirements.txt

# 청킹 전략 비교
python examples/compare_chunking.py

# 하이브리드 검색 실험
python examples/compare_search.py
```

### 3단계: 벡터 데이터베이스 (1-2시간)

```bash
cd 03-vector-databases
pip install -r requirements.txt

# ChromaDB 예제 (가장 간단)
python chromadb/basic_example.py

# 성능 비교
python comparison/speed_benchmark.py
```

## 📖 학습 경로

### 초급 (1-2일)
1. **01-basic-rag** 전체 실습
2. 샘플 문서 추가하고 재실행
3. 다양한 질문으로 테스트
4. 코드 읽고 이해하기

### 중급 (3-5일)
1. **02-advanced-rag** 각 기법 실습
2. 청킹 전략별 성능 비교
3. 하이브리드 검색 구현
4. 리랭킹 효과 측정

### 고급 (1주일)
1. **03-vector-databases** 전체 실습
2. 각 벡터 DB 특징 파악
3. 실전 프로젝트에 적용
4. 성능 최적화 및 튜닝

## 🛠️ 기술 스택

### 핵심 라이브러리
- **Sentence Transformers**: 임베딩 생성
- **Transformers**: LLM 모델
- **NumPy**: 벡터 연산
- **PyTorch**: 딥러닝 프레임워크

### 벡터 데이터베이스
- **FAISS**: Meta의 고성능 벡터 검색
- **ChromaDB**: AI 네이티브 벡터 DB
- **Pinecone**: 관리형 클라우드 서비스
- **Weaviate**: 오픈소스 벡터 검색 엔진
- **Qdrant**: Rust 기반 고성능 DB

### 고급 기법
- **Rank-BM25**: 전통적 텍스트 검색
- **LangChain**: RAG 파이프라인 구축
- **RAGAS**: RAG 평가 프레임워크

## 📊 주요 개념

### RAG란?
```
사용자 질문
    ↓
[1. 검색 단계]
질문 임베딩 → 벡터 검색 → 관련 문서 추출
    ↓
[2. 생성 단계]
문서 + 질문 → LLM → 답변 생성
```

### 왜 RAG가 필요한가?
- ✅ **최신 정보**: LLM 학습 데이터 이후의 정보 활용
- ✅ **도메인 지식**: 회사/분야 특화 지식 반영
- ✅ **환각 감소**: 문서 기반 답변으로 정확도 향상
- ✅ **추적 가능**: 답변의 출처 확인 가능

### RAG vs Fine-tuning
| 특징 | RAG | Fine-tuning |
|------|-----|-------------|
| 비용 | 낮음 | 높음 |
| 업데이트 | 즉시 | 재학습 필요 |
| 정확도 | 높음 (출처 기반) | 높음 (모델 내재화) |
| 유연성 | 높음 | 낮음 |
| 사용 사례 | 지식 기반 QA | 스타일/톤 학습 |

## 💡 실전 팁

### 성능 최적화
1. **청킹 크기**: 256-512 토큰 권장
2. **Overlap**: 10-20% 유지
3. **Top-K**: 초기 20-50개, 리랭킹 후 3-5개
4. **임계값**: 유사도 0.6-0.7 이상 필터링

### 프롬프트 엔지니어링
```python
prompt = """
Context:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- If the answer is not in the context, say "정보가 충분하지 않습니다"
- Cite the source document when possible

Answer:
"""
```

### 디버깅 체크리스트
- [ ] 임베딩 모델이 한국어를 지원하는가?
- [ ] 문서가 너무 길어 잘리지 않았는가?
- [ ] 검색 결과가 실제로 관련이 있는가?
- [ ] LLM이 컨텍스트를 무시하고 있지 않은가?
- [ ] 프롬프트가 명확한가?

## 🔧 문제 해결

### CUDA 오류
```python
# GPU 없이 CPU로 실행
device = -1  # transformers pipeline에서
```

### 메모리 부족
```python
# 배치 크기 줄이기
embeddings = model.encode(texts, batch_size=16)  # 기본 32
```

### 임베딩 모델 다운로드 느림
```python
# 작은 모델 사용
model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB
```

## 📚 학습 자료

### 논문
- [RAG 원본 논문](https://arxiv.org/abs/2005.11401) - Lewis et al., 2020
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [REALM](https://arxiv.org/abs/2002.08909)

### 문서
- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)

### 블로그/튜토리얼
- [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [LlamaIndex Docs](https://docs.llamaindex.ai/)

## 🤝 기여하기

이슈와 풀 리퀘스트를 환영합니다!

### 기여 가이드
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 라이선스

MIT License - 자유롭게 사용, 수정, 배포할 수 있습니다.

## 🙋 질문 및 피드백

- **Issues**: 버그 리포트 및 기능 제안
- **Discussions**: 질문 및 토론
- **Email**: [이메일 주소]

## 🗺️ 로드맵

### v1.0 (현재)
- ✅ 기본 RAG 구현
- ✅ 고급 RAG 기법
- ✅ 벡터 DB 비교

### v1.1 (예정)
- [ ] 멀티모달 RAG (이미지 + 텍스트)
- [ ] 그래프 RAG
- [ ] 에이전트 기반 RAG

### v2.0 (계획)
- [ ] 실전 프로젝트 템플릿
- [ ] 프로덕션 배포 가이드
- [ ] 성능 최적화 가이드

## 📈 학습 진행 체크리스트

### Week 1: 기초
- [ ] 01-basic-rag 완료
- [ ] RAG 동작 원리 이해
- [ ] 간단한 질의응답 시스템 구현

### Week 2: 심화
- [ ] 02-advanced-rag 완료
- [ ] 청킹 전략 실험
- [ ] 하이브리드 검색 구현

### Week 3: 응용
- [ ] 03-vector-databases 완료
- [ ] 3개 이상의 벡터 DB 비교
- [ ] 최종 프로젝트 설계

### Week 4: 프로젝트
- [ ] 실제 데이터로 RAG 시스템 구축
- [ ] 성능 평가 및 최적화
- [ ] 배포 준비

## 🎓 추천 학습 순서

1. **입문자** (프로그래밍 경험 있음)
   - 01-basic-rag → 문서 읽기 → 코드 실행 → 이해하기
   - 2-3일 소요

2. **중급자** (ML 기초 지식 있음)
   - 01-basic-rag (빠르게) → 02-advanced-rag (집중) → 03-vector-databases
   - 1주일 소요

3. **고급자** (RAG 경험 있음)
   - 02-advanced-rag, 03-vector-databases → 심화 학습 → 프로젝트 적용
   - 2-3일 소요

---

**Happy Learning! 🎉**

궁금한 점이 있다면 언제든지 Issues에 질문을 남겨주세요.

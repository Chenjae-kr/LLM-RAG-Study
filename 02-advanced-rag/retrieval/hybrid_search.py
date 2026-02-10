"""
하이브리드 검색 (Hybrid Search)
Dense Retrieval (임베딩 기반)과 Sparse Retrieval (BM25)을 결합합니다.
"""

import numpy as np
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridSearcher:
    def __init__(self, documents: List[str], embedding_model: str = "all-MiniLM-L6-v2"):
        """
        하이브리드 검색기 초기화

        Args:
            documents: 검색 대상 문서 리스트
            embedding_model: 사용할 임베딩 모델
        """
        self.documents = documents

        # Dense Retrieval 준비
        print("임베딩 모델 로딩 중...")
        self.encoder = SentenceTransformer(embedding_model)
        print("문서 임베딩 생성 중...")
        self.doc_embeddings = self.encoder.encode(documents, show_progress_bar=True)

        # Sparse Retrieval 준비 (BM25)
        print("BM25 인덱스 생성 중...")
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        print("하이브리드 검색기 준비 완료!")

    def dense_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Dense Retrieval (임베딩 기반 검색)

        Returns:
            (문서 인덱스, 유사도 점수) 튜플의 리스트
        """
        query_embedding = self.encoder.encode([query])[0]

        # 코사인 유사도 계산
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = self.doc_embeddings / np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        similarities = np.dot(doc_norms, query_norm)

        # 상위 K개 선택
        top_indices = np.argsort(-similarities)[:top_k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]

        return results

    def sparse_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Sparse Retrieval (BM25 기반 검색)

        Returns:
            (문서 인덱스, BM25 점수) 튜플의 리스트
        """
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        # 상위 K개 선택
        top_indices = np.argsort(-scores)[:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices]

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0
    ) -> List[Tuple[int, float, dict]]:
        """
        하이브리드 검색: Dense와 Sparse 결과를 결합

        Args:
            query: 검색 쿼리
            top_k: 최종 반환할 문서 수
            alpha: Dense 검색 가중치 (0~1, Sparse는 1-alpha)
            dense_weight: Dense 점수 스케일링 가중치
            sparse_weight: Sparse 점수 스케일링 가중치

        Returns:
            (문서 인덱스, 최종 점수, 세부 점수) 튜플의 리스트
        """
        # 각각 더 많은 후보 가져오기
        candidate_k = top_k * 3

        dense_results = self.dense_search(query, candidate_k)
        sparse_results = self.sparse_search(query, candidate_k)

        # 점수 정규화
        dense_scores = {idx: score for idx, score in dense_results}
        sparse_scores = {idx: score for idx, score in sparse_results}

        # 점수 정규화 (0~1 범위로)
        if dense_scores:
            max_dense = max(dense_scores.values())
            min_dense = min(dense_scores.values())
            if max_dense > min_dense:
                dense_scores = {k: (v - min_dense) / (max_dense - min_dense) * dense_weight
                               for k, v in dense_scores.items()}

        if sparse_scores:
            max_sparse = max(sparse_scores.values())
            min_sparse = min(sparse_scores.values())
            if max_sparse > min_sparse:
                sparse_scores = {k: (v - min_sparse) / (max_sparse - min_sparse) * sparse_weight
                                for k, v in sparse_scores.items()}

        # 결합된 점수 계산
        all_indices = set(dense_scores.keys()) | set(sparse_scores.keys())
        combined_scores = []

        for idx in all_indices:
            d_score = dense_scores.get(idx, 0.0)
            s_score = sparse_scores.get(idx, 0.0)
            final_score = alpha * d_score + (1 - alpha) * s_score

            combined_scores.append((
                idx,
                final_score,
                {
                    'dense': d_score,
                    'sparse': s_score,
                    'combined': final_score
                }
            ))

        # 최종 점수로 정렬하여 상위 K개 반환
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores[:top_k]

    def search_with_results(self, query: str, top_k: int = 5, alpha: float = 0.5):
        """
        검색 결과를 문서 텍스트와 함께 반환

        Returns:
            검색 결과 리스트 (문서 텍스트, 점수, 세부 정보)
        """
        results = self.hybrid_search(query, top_k, alpha)

        formatted_results = []
        for idx, score, details in results:
            formatted_results.append({
                'document': self.documents[idx],
                'score': score,
                'dense_score': details['dense'],
                'sparse_score': details['sparse']
            })

        return formatted_results


if __name__ == "__main__":
    # 샘플 문서
    documents = [
        "인공지능은 기계가 인간의 지능을 모방하는 기술입니다.",
        "머신러닝은 데이터로부터 학습하는 AI의 하위 분야입니다.",
        "딥러닝은 신경망을 사용하는 머신러닝 기법입니다.",
        "자연어 처리는 컴퓨터가 인간의 언어를 이해하는 기술입니다.",
        "RAG는 검색과 생성을 결합한 AI 기법입니다.",
        "벡터 데이터베이스는 임베딩을 효율적으로 저장합니다.",
        "트랜스포머는 어텐션 메커니즘을 사용하는 모델입니다.",
        "GPT는 텍스트 생성에 특화된 대형 언어 모델입니다.",
    ]

    # 검색기 초기화
    searcher = HybridSearcher(documents)

    # 검색 테스트
    query = "언어 모델에 대해 알려주세요"

    print(f"\n=== 쿼리: '{query}' ===\n")

    print("1. Dense Search (임베딩 기반)")
    dense_results = searcher.dense_search(query, top_k=3)
    for idx, score in dense_results:
        print(f"  점수 {score:.4f}: {documents[idx]}")

    print("\n2. Sparse Search (BM25 기반)")
    sparse_results = searcher.sparse_search(query, top_k=3)
    for idx, score in sparse_results:
        print(f"  점수 {score:.4f}: {documents[idx]}")

    print("\n3. Hybrid Search (alpha=0.5)")
    hybrid_results = searcher.search_with_results(query, top_k=3, alpha=0.5)
    for result in hybrid_results:
        print(f"  최종 {result['score']:.4f} (dense: {result['dense_score']:.4f}, sparse: {result['sparse_score']:.4f})")
        print(f"    {result['document']}")

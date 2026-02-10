"""
고정 크기 청킹 (Fixed-size Chunking)
토큰 수 또는 문자 수를 기준으로 문서를 일정한 크기로 분할합니다.
"""

from typing import List


def chunk_by_characters(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    문자 수 기반 고정 크기 청킹

    Args:
        text: 원본 텍스트
        chunk_size: 청크당 문자 수
        overlap: 청크 간 겹치는 문자 수

    Returns:
        청크 리스트
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_by_sentences(text: str, sentences_per_chunk: int = 5, overlap: int = 1) -> List[str]:
    """
    문장 수 기반 고정 크기 청킹

    Args:
        text: 원본 텍스트
        sentences_per_chunk: 청크당 문장 수
        overlap: 청크 간 겹치는 문장 수

    Returns:
        청크 리스트
    """
    # 간단한 문장 분리 (실제로는 nltk나 spacy 사용 권장)
    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]

    chunks = []
    i = 0

    while i < len(sentences):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        chunk = ' '.join(chunk_sentences)
        chunks.append(chunk)
        i += sentences_per_chunk - overlap

    return chunks


def chunk_with_metadata(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[dict]:
    """
    메타데이터를 포함한 청킹

    Returns:
        각 청크의 텍스트와 메타데이터를 포함한 딕셔너리 리스트
    """
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        chunks.append({
            'id': chunk_id,
            'text': chunk_text,
            'start': start,
            'end': end,
            'length': len(chunk_text)
        })

        start += chunk_size - overlap
        chunk_id += 1

    return chunks


if __name__ == "__main__":
    # 테스트
    sample_text = """
    인공지능은 현대 기술의 핵심 분야입니다. 머신러닝과 딥러닝 기술의 발전으로 많은 혁신이 일어나고 있습니다.
    자연어 처리 분야에서는 트랜스포머 아키텍처가 큰 성공을 거두었습니다. GPT와 BERT 같은 모델들이 등장했습니다.
    컴퓨터 비전 분야에서도 CNN과 Vision Transformer가 활용됩니다. 이미지 인식 정확도가 크게 향상되었습니다.
    """.strip()

    print("=== 문자 기반 청킹 ===")
    char_chunks = chunk_by_characters(sample_text, chunk_size=100, overlap=20)
    for i, chunk in enumerate(char_chunks):
        print(f"\nChunk {i}: {chunk}")

    print("\n\n=== 문장 기반 청킹 ===")
    sent_chunks = chunk_by_sentences(sample_text, sentences_per_chunk=2, overlap=1)
    for i, chunk in enumerate(sent_chunks):
        print(f"\nChunk {i}: {chunk}")

    print("\n\n=== 메타데이터 포함 청킹 ===")
    meta_chunks = chunk_with_metadata(sample_text, chunk_size=100, overlap=20)
    for chunk in meta_chunks:
        print(f"\nChunk {chunk['id']}: {chunk['text'][:50]}... (length: {chunk['length']})")

# chunk_size
# overlap
# parent-child 함수
# basic_split
# parent_child_split

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict


# =========================
# Basic Splitter
# =========================
def basic_split(
    documents: List[str],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[str]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    return chunks


# =========================
# Parent-Child Splitter
# =========================
def parent_child_split(
    documents: List[str],
    parent_chunk_size: int = 1500,
    parent_overlap: int = 200,
    child_chunk_size: int = 400,
    child_overlap: int = 50,
) -> Dict[str, List]:

    # 1️⃣ Parent 생성
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap
    )

    parents = parent_splitter.split_documents(documents)

    # 2️⃣ Child 생성
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap
    )

    child_chunks = []
    parent_docs = []

    for idx, parent in enumerate(parents):
        parent_id = f"parent_{idx}"

        # parent 메타데이터에 ID 추가
        parent.metadata["id"] = parent_id
        parent_docs.append(parent)

        # child 생성
        children = child_splitter.split_documents([parent])

        for child in children:
            child.metadata["parent_id"] = parent_id
            child_chunks.append(child)

    return {
        "parents": parent_docs,
        "children": child_chunks
    }


# =========================
# 전략 선택 함수
# =========================
def get_chunks(documents: List[str], strategy: str = "basic"):

    if strategy == "basic":
        return basic_split(documents)

    elif strategy == "parent_child":
        return parent_child_split(documents)

    else:
        raise ValueError("Unknown strategy")

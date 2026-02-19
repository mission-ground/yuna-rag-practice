from sentence_transformers import SentenceTransformer
import chromadb
from splitter import get_chunks
from langchain_community.document_loaders import TextLoader

# =============================
# 모델 로드
# =============================
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# =============================
# 문서 로드 + Parent-Child 분리
# =============================
loader = TextLoader("data.txt", encoding="utf-8")
documents = loader.load()

result = get_chunks(documents, strategy="parent_child")

parent_docs = result["parents"]
child_chunks = result["children"]

print("Parent 수:", len(parent_docs))
print("Child 수:", len(child_chunks))

# =============================
# Parent Docstore 구성
# =============================
docstore = {}

for parent in parent_docs:
    parent_id = parent.metadata["id"]
    docstore[parent_id] = parent.page_content

# =============================
# Child 벡터용 데이터 준비
# =============================
child_contents = []
child_ids = []
child_metadatas = []

for i, chunk in enumerate(child_chunks):
    chunk_id = f"chunk_{i}"

    child_ids.append(chunk_id)
    child_contents.append(chunk.page_content)

    child_metadatas.append({
        "parent_id": chunk.metadata["parent_id"]
    })

# =============================
# 임베딩 (Child만)
# =============================
embeddings = model.encode(child_contents, normalize_embeddings=True)

# =============================
# Chroma 저장
# =============================
client = chromadb.Client()

try:
    client.delete_collection("chunks_docs")
except:
    pass

collection = client.create_collection(name="chunks_docs")

collection.add(
    documents=child_contents,
    embeddings=embeddings.tolist(),
    ids=child_ids,
    metadatas=child_metadatas
)

print("DB 문서 수:", collection.count())

def search(query: str, k_chunk: int = 3):

    query_embedding = model.encode([query], normalize_embeddings=True)

    # 1️⃣ Child 검색
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k_chunk
    )

    if not results["ids"][0]:
        return []

    # 2️⃣ 가장 유사한 child의 parent_id 가져오기
    best_parent_id = results["metadatas"][0][0]["parent_id"]
    best_distance = results["distances"][0][0]

    print("Best Parent:", best_parent_id)
    print("Distance:", best_distance)

    # 3️⃣ Parent 복원
    parent_text = docstore.get(best_parent_id)

    return parent_text


# 테스트
if __name__ == "__main__":
    print(search("RAG의 한계는 무엇인가요?"))

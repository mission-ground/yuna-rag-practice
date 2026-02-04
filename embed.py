from sentence_transformers import SentenceTransformer
import json
import chromadb
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")



with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)


# texts = [c["text"] for c in chunks]
# titles = [c["title"] for c in chunks]
ids = [c["id"] for c in chunks]

contents = [
    f"{c['title']}: {c['text']}" if 'title' in c else c['text']
    for c in chunks
]

embeddings = model.encode(contents, normalize_embeddings=True) # 정규화된 벡터?

# print(type(embeddings))
# print(type(embeddings[0]))
# print(len(embeddings[0]))
# print(embeddings[0][:10])  # 앞부분만

# print(cosine_similarity([embeddings[0]], [embeddings[1]]))

metadatas = []

for c in chunks:
    meta = {
        "type": "anchor" if "parent_id" not in c else "child"
    }
    if "parent_id" in c:
        meta["parent_id"] = c["parent_id"]
    metadatas.append(meta)

client = chromadb.Client()
#collection = client.create_collection(name="chunks_docs")
collection = client.get_or_create_collection(name="chunks_docs")

if collection.count() == 0:
    collection.add(
        documents=contents,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=metadatas
    )

# print(collection.count())
# print(collection.peek())


# 앵커의 children 리스트
anchor_map = defaultdict(list)

for c in chunks:
    parent_id = c.get("parent_id")  # anchor id
    if parent_id:
        anchor_map[parent_id].append(c["id"])


def search(query: str, k_anchor: int = 1, k_chunk: int = 7):

    # query = "osi의 역사는 어떻게 진행되나요?"
    query_embedding = model.encode([query], normalize_embeddings=True)

    anchor_results = collection.query( #유사한 것 찾기
        query_embeddings=query_embedding.tolist(),
        n_results=k_anchor,
        where={"type":"anchor"}
    )
    # print(anchor_results)
      # 가장 관련 있는 앵커 ID 가져오기

    if not anchor_results["ids"][0]:
        return []

    anchor_id = anchor_results["ids"][0][0]
    print(f"Top Anchor ID: {anchor_id}")

     # 2️⃣ 하위 청크 검색
    child_ids = anchor_map.get(anchor_id, [])

    print("DEBUG children:", anchor_map.get(anchor_id))

    best_distance = anchor_results["distances"][0][0]

    if best_distance <= 0.8 and child_ids:
        chunk_results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k_chunk,
            ids=child_ids
            # where={"id": {"$in": child_ids}}
        )
        # print("◎관련 하위 청크:")
        # for doc in chunk_results["documents"]:
        #     print("-", doc)

        return chunk_results["documents"][0]
    
   
   # 없을경우 전체에서 k 추출
   
    fallback_results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3,
        where={"type":"child"}
    )
    print(fallback_results)
    return fallback_results["documents"][0]
    
    # 첫 번째 청크 반환

    # return results["documents"][0]

search("osi 역사??")


# for i, doc in enumerate(results["documents"][0]):
#     print(f"\n[{i+1}번 결과]")
#     print(doc)

###임베딩 저장####

# import numpy as np

# np.save("embeddings.npy", embeddings)
# loaded = np.load("embeddings.npy")

# print(np.allclose(embeddings[0], loaded[0])) # True

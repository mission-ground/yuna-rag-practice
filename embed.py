from sentence_transformers import SentenceTransformer
import json
import chromadb
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")





with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)


texts = [c["text"] for c in chunks]
ids = [c["id"] for c in chunks]

embeddings = model.encode(texts, normalize_embeddings=True) # 정규화된 벡터?

# print(type(embeddings))
# print(type(embeddings[0]))
# print(len(embeddings[0]))
# print(embeddings[0][:10])  # 앞부분만

# print(cosine_similarity([embeddings[0]], [embeddings[1]]))

client = chromadb.Client()
#collection = client.create_collection(name="chunks_docs")
collection = client.get_or_create_collection(name="chunks_docs")

if collection.count() == 0:
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=ids
    )

# print(collection.count())

def search(query: str, k: int = 8):

    # query = "osi의 역사는 어떻게 진행되나요?"
    query_embedding = model.encode([query], normalize_embeddings=True)

    results = collection.query( #유사한 것 찾기
        query_embeddings=query_embedding.tolist(),
        n_results=k
    )
    print(results)
    return results["documents"][0]

# for i, doc in enumerate(results["documents"][0]):
#     print(f"\n[{i+1}번 결과]")
#     print(doc)

###임베딩 저장####

# import numpy as np

# np.save("embeddings.npy", embeddings)
# loaded = np.load("embeddings.npy")

# print(np.allclose(embeddings[0], loaded[0])) # True

import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import chromadb

def embed_data():
    df = pd.read_csv("data/emoji_dataset.csv")
    texts = df["text"].tolist()
    emojis = df["emoji"].tolist()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    client = chromadb.PersistentClient(path="vector_store")

    collection = client.get_or_create_collection(name="emoji_vectors")

    docs = [
        {"id": str(i), "document": texts[i], "embedding": embeddings.embed_query(texts[i]), "metadata": {"emoji": emojis[i]}}
        for i in range(len(texts))
    ]

    for doc in docs:
        collection.add(
            ids=[doc["id"]],
            documents=[doc["document"]],
            embeddings=[doc["embedding"]],
            metadatas=[doc["metadata"]]
        )

    print("[✅] Embeddings stored with Chroma (langchain_chroma version)")

if __name__ == "__main__":
    embed_data()

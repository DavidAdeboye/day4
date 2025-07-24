import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

def embed_data():
    df = pd.read_csv("data/emoji_dataset.csv")
    texts = df["text"].tolist()
    emojis = df["emoji"].tolist()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vectorstore = Chroma.from_texts(
        texts,
        embedding=embeddings,
        metadatas=[{"emoji": e} for e in emojis],
        collection_name="emoji_vectors",
        persist_directory="vector_store"
    )
    
    vectorstore.persist()
    print("[✅] Embeddings stored successfully.")

if __name__ == "__main__":
    embed_data()

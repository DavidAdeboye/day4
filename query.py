from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import chromadb

def predict_emoji(text, k=1):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    client = chromadb.PersistentClient(path="vector_store")
    collection = client.get_collection(name="emoji_vectors")

    query_embedding = embeddings.embed_query(text)

    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    if results["metadatas"] and len(results["metadatas"][0]) > 0:
        return [meta["emoji"] for meta in results["metadatas"][0]]
    else:
        return ["🤷‍♂️"]

# Test
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            break
        print("Emoji:", predict_emoji(user_input)[0])

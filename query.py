from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

def predict_emoji(text, k=1):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name="emoji_vectors",
        persist_directory="vector_store"
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(text)

    if results:
        return [doc.metadata["emoji"] for doc in results]
    else:
        return ["🤷‍♂️"]

# Test
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            break
        print("Emoji:", predict_emoji(user_input)[0])

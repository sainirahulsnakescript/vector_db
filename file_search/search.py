from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

def load_vectorstore():
    vectorstore_path = 'vectorstore/db_faiss'
    if os.path.exists(vectorstore_path):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        return FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    else:
        raise FileNotFoundError(f"No vector store found at {vectorstore_path}")

def retrieve_relevant_chunks(query, vectorstore, top_k=1):
    # Use FAISS to retrieve the most relevant document chunks
    results = vectorstore.similarity_search(query, k=top_k)
    return results

def answer_question(query, top_k=1):
    vectorstore = load_vectorstore()
    relevant_chunks = retrieve_relevant_chunks(query, vectorstore, top_k)
    
    # Simple answer generation: concatenate the retrieved chunks
    answer = "\n".join([chunk.page_content for chunk in relevant_chunks])
    return answer

# Example usage
if __name__ == "__main__":
    question = "What is a ‘Substantial Benefit’ in the context of co-borrowers?"
    answer = answer_question(question)
    print(answer)

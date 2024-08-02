from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
# Function to load the vector store
def load_vectorstore():
    vectorstore_path = 'vectorstore/db_faiss'
    if os.path.exists(vectorstore_path):
        embeddings = OpenAIEmbeddings()
        return FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    else:
        raise FileNotFoundError(f"No vector store found at {vectorstore_path}")


def get_answer(question, vectorstore):
    # Initialize the language model with the API key
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    
    # Create the RetrievalQA chain
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Ensure this chain type is appropriate
        retriever=retriever
    )
    
    # Run the QA chain with the question
    answer = qa_chain.invoke({"query": question})
    return answer




# Example usage
if __name__ == "__main__":
    vectorstore = load_vectorstore()
    question = "What specific restrictions apply to the security property in the Category 3 ‘Mining’ postcode for both owner-occupied and investment purposes"
    answer = get_answer(question, vectorstore)
    print(answer)

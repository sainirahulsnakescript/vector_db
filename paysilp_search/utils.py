import os
import platform
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def get_folder_list(folder_path):
    items = os.listdir(folder_path)
    folders = [item for item in items if os.path.isdir(os.path.join(folder_path,item))]
    return folders

def get_specific_folder_list(folder_path):
    items = os.listdir(folder_path)
    files = [item for item in items if os.path.isfile(os.path.join(folder_path,item))]
    return files

def clear_terminal():
    # Determine the operating system
    current_os = platform.system()
    
    if current_os == "Windows":
        os.system('cls')  # Windows command to clear the terminal
    else:
        os.system('clear')  # Unix/Linux/Mac command to clear the terminal

# Function to load the vector store
def load_vectorstore(folder_path,index):
    vectorstore_path = f'vectorstore/payslip_db_faiss/{folder_path}'
    if os.path.exists(vectorstore_path):
        embeddings = OpenAIEmbeddings()
        return FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True,index_name=index)
    else:
        raise FileNotFoundError(f"No vector store found at {vectorstore_path}")
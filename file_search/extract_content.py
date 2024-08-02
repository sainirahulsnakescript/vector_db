import os 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

def generate_unique_id(existing_ids):
    """Generate a unique ID not in the existing set of IDs."""
    while True:
        new_id = str(uuid4())
        if new_id not in existing_ids:
            print(new_id)
            return new_id



def process_file_data(file_path:list):
    try:
        # Check if each item in the list is a valid .txt file path
        for path in file_path:
            if not isinstance(path, str):
                raise ValueError(f"Expected a string for file path, but got {type(path)}")
            if not path.lower().endswith('.txt'):
                raise ValueError(f"File path does not have a .txt extension: {path}")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"The file path does not exist: {path}")

        embeddings = OpenAIEmbeddings()
        vectorstore_path = 'vectorstore/db_faiss'
        if os.path.exists(vectorstore_path):
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            existing_ids = set(doc.metadata['id'] for doc in vectorstore.get_all_documents())

        else:
            vectorstore = None
            existing_ids = set()
        data = []
        # Load documents from DOCX files
        for file in file_path:
            filename = os.path.basename(file)
            if filename.endswith('.txt'):
                loader = TextLoader(file)
                document = loader.load()
                document[0].metadata['id'] = generate_unique_id(existing_ids)
                data.extend(document)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(data)
        if vectorstore is None:
            vectorstore = FAISS.from_documents(splits,embeddings)
        text_id_list =  vectorstore.add_documents(splits)
        vectorstore.save_local(vectorstore_path)
    except Exception as e:
        print(f"exception error: {e}")




file_list = [
    'file_search/file/Firstmac Lending Policy.txt',
    'file_search/file/NAB Policy.txt',
]
process_file_data(file_list)
# process_files(file_list)
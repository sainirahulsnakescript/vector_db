import logging.config
import os 
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

def get_folder_list(folder_path):
    items = os.listdir(folder_path)
    folders = [item for item in items if os.path.isdir(os.path.join(folder_path,item))]
    return folders

def get_specific_folder_list(folder_path):
    items = os.listdir(folder_path)
    files = [item for item in items if os.path.isfile(os.path.join(folder_path,item))]
    return files


def process_payslip(folder_path,file):
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore_path = f'vectorstore/payslip_db_faiss/{folder_path}'
        vectorstore = None
        data = []
        file_path = os.path.join(folder_path,file)
        # Load documents from DOCX files
        filename = os.path.basename(file_path)
        if filename.endswith('.txt'):
            loader = TextLoader(file_path)
            document = loader.load()
            data.extend(document)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(data)
        if vectorstore is None:
            vectorstore = FAISS.from_documents(splits,embeddings)
        text_id_list =  vectorstore.add_documents(splits)
        # Extract the filename without the extension
        index_file_name = os.path.splitext(filename)[0]
        vectorstore.save_local(vectorstore_path,index_name=index_file_name)
        os.remove(file_path)
    except Exception as e:
        print(f"exception error: {e}")


import fitz
def convert_pdf_to_text(file_path):
    # Open the PDF file
    pdf_document = fitz.open(file_path)
    
    # Initialize an empty string to store the extracted text
    text = ""
    
    # Iterate through each page
    for page_num in range(len(pdf_document)):
        # Get the page
        page = pdf_document.load_page(page_num)
        # Extract text from the page
        text += page.get_text()
    
    # Close the PDF document
    pdf_document.close()
    
    # Create the output file path by changing the extension to .txt
    base_name = os.path.splitext(file_path)[0]
    output_file_path = f"{base_name}.txt"
    
    # Save the extracted text to the .txt file
    with open(output_file_path, "w", encoding="utf-8") as text_file:
        text_file.write(text)
    
    return output_file_path
        



payslip_folder_path = f'paysilp_search/payslip_floder'
folder_list = get_folder_list(payslip_folder_path)

for folder in folder_list:
    single_folder_path = os.path.join(payslip_folder_path,folder)
    files = get_specific_folder_list(single_folder_path)
    for file in files:
        file_document = os.path.join(single_folder_path,file)
        text_file_path =  convert_pdf_to_text(file_document)
        text_file_name  = os.path.basename(text_file_path)
        process_payslip(single_folder_path,text_file_name)
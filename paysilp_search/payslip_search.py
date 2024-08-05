from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import create_retrieval_chain
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Payslip Search')
from dotenv import load_dotenv
load_dotenv()
from utils import(
    clear_terminal,
    get_folder_list,
    get_specific_folder_list,
    load_vectorstore
)


def get_folder_names_from_gpt(question,folder_list):
    # Initialize the ChatGPT model
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=150,  # Adjust the max_tokens as needed
        timeout=None,
        max_retries=2
    )
    
    folder_list_formatted = "\n".join([f"{i + 1}. {folder}" for i, folder in enumerate(folder_list)])
    prompt = (
        f"Given the following question, identify which folder from the list "
        f"contains the most relevant information. Please respond with a dictionary "
        f"format where the key is 'folder_name' and the value is the name of the folder. "
        f"\n\nQuestion: {question}\n\nList of folders:\n"
        f"{folder_list_formatted}\n\n"
        f"Example response format:\n{{'folder_name': 'FolderName'}}\n\n"
        f"Response:"
    )
    response = llm(prompt)
    folder_dict = eval(response.content)
    if type(folder_dict) is dict and folder_dict.get('folder_name') and folder_dict.get('folder_name') is not None:
        return folder_dict.get('folder_name')
    return get_folder_names_from_gpt(question,folder_list)
    


def get_file_names_from_gpt(question,file_list):
    # Initialize the ChatGPT model
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=150,  # Adjust the max_tokens as needed
        timeout=None,
        max_retries=2
    )
    
    file_list_formatted = "\n".join([f"{i + 1}. {folder}" for i, folder in enumerate(file_list)])
    prompt = (
        f"Given the following question, identify which file from the list "
        f"contains the most relevant information. If a file is relevant to the question, "
        f"return a dictionary where the key is 'file_name' and the value is the name of the file. "
        f"If none of the files are relevant, respond with 'None'. "
        f"\n\nQuestion: {question}\n\nList of files:\n"
        f"{file_list_formatted}\n\n"
        f"Note: When matching names, prioritize full name matches. If multiple variations of first names are present, "
        f"consider them. If no full name match is found, match based on the last name or middle name as applicable. "
        f"Ensure that similar variations of names are treated as equivalent."
        f"\n\nExample response format:\n{{'file_name': 'FileName'}} or {{'file_name': 'None'}}\n\n"
        f"Response:"
        )
    response = llm(prompt)
    file_dict = eval(response.content)
    if type(file_dict) is dict and file_dict.get('file_name'):
        if file_dict.get('file_name').lower() != 'none':
            return file_dict.get('file_name')
        else:
            raise FileNotFoundError(f'File Not Found')
    return get_file_names_from_gpt(question,file_list)

    
def find_folder_by_name(root_folder, target_folder_name):
    """
    Recursively search for a folder by name within the root folder and its subdirectories.
    
    Parameters:
    root_folder (str): The path to the root folder.
    target_folder_name (str): The name of the folder to search for.
    
    Returns:
    str or None: The path to the folder if found, otherwise None.
    """
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if target_folder_name in dirnames:
            return os.path.join(dirpath, target_folder_name)
    return None


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


def get_vector_db_index(query):
    parent_folder ='vectorstore/payslip_db_faiss/paysilp_search/payslip_floder'
    folder_list = get_folder_list(parent_folder)
    folder_name = get_folder_names_from_gpt(query,folder_list)
    logger.info(f'Folder Name Get from Gpt: {folder_name}')
    folder_path = find_folder_by_name(parent_folder,folder_name)
    logger.info(f'folder path: {folder_path}')
    file_list = get_specific_folder_list(folder_path)
    file_name = get_file_names_from_gpt(query,file_list)
    logger.info(f'file name get from gpt: {file_name}')
    vector_database_path = 'vectorstore/payslip_db_faiss'
    complete_path_db = folder_path
    if os.path.exists(complete_path_db):
        logger.info(f'Folder path in database Exits at: {complete_path_db}')
        base_file_name = os.path.splitext(file_name)[0]
        pkl_path = os.path.join(complete_path_db,base_file_name+'.pkl')
        faiss_path = os.path.join(complete_path_db,base_file_name+'.faiss')
        if os.path.exists(pkl_path) and os.path.exists(faiss_path):
            logger.info(f'Succesfully Found index of File "{file_name} \n Index of file {os.path.splitext(file_name)[0]}"')
            return {"index":os.path.splitext(file_name)[0],"folder_path":folder_path}
    else:
        logger.error(f'Folder in database not foundat : {complete_path_db}')

def question_answer():
    try:
        question = f'Tell me about  payslips for client Ahmet and give me personal details'
        db_index = get_vector_db_index(question)
        vectorstore = load_vectorstore(folder_path=db_index['folder_path'],index=db_index['index'])
        answer = get_answer(question, vectorstore)
        clear_terminal()
        # Extract the result from the answer
        result = answer.get('result', 'No result found.')
        
        # Print the formatted result
        print("\n--- Payslip Information ---")
        print(result)
        print("\n-----------------------------")
    except FileNotFoundError as e:
        logger.error(e)

question_answer()

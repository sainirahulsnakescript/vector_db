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
        f"format where the key is 'folder_names' and the value is a list of relevant folder names. "
        f"\n\nQuestion: {question}\n\nList of folders:\n"
        f"{folder_list_formatted}\n\n"
        f"Example response format:\n{{'folder_names': ['FolderName1', 'FolderName2']}}\n\n"
        f"Response:"
    )
    response = llm.invoke(prompt)
    folder_dict = eval(response.content)
    if type(folder_dict) is dict and folder_dict.get('folder_names') and folder_dict.get('folder_names') is not None:
        return folder_dict.get('folder_names')
    else: 
        return folder_dict.get('folder_names')
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
    
    # Format the file list
    file_list_formatted = "\n".join([
        f"{i + 1}. {folder_name}:\n"
        f"   folder_path: {details['path']}\n"
        f"   file_in_folder: {details['file_name']}"
        for i, (folder_name, details) in enumerate(file_list.items())
    ])

    prompt = (
        f"Given the following question, identify which files from the list "
        f"contains the most relevant information. Please respond with a dictionary "
        f"format where the key is 'file_names' and the value is a list of relevant file names(only .pkl file). "
        f"\n\nQuestion: {question}\n\nList of files:\n"
        f"{file_list_formatted}\n\n"
        f"Example response format:\n{{'file_names':['fileName1', 'FolderName2']}}\n\n"
        f"Your response going eval(response.content)"
        f"Response should be like {{'file_names':['FileName1','FileName2']}} only dict not other content"
        f"Response: {{'file_name':'[list of file]'}}"
    )
    response = llm.invoke(prompt)
    try:
        file_dict = eval(response.content)
        if type(file_dict) is dict and file_dict.get('file_names'):
            if type(file_dict['file_names']) is list:
                return file_dict.get('file_names')
            else:
                raise FileNotFoundError(f'File Not Found')
    except:
        get_file_names_from_gpt(question,file_list)
    return get_file_names_from_gpt(question,file_list)

    
def find_folder_by_name(root_folder, target_folder_names):
    """
    Recursively search for folders by names within the root folder and its subdirectories.
    
    Parameters:
    root_folder (str): The path to the root folder.
    target_folder_names (list of str): The list of folder names to search for.
    
    Returns:
    dict: A dictionary where the keys are the folder names and the values are their paths. 
          If a folder is not found, its value will be None.
    """
    result = {folder_name: None for folder_name in target_folder_names}
    folder_path = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for folder_name in target_folder_names:
            if folder_name in dirnames:
                result[folder_name] = {'path':os.path.join(dirpath, folder_name)}
    
    return result



def find_file_by_name(root_folder, target_file_names):
    """
    Recursively search for folders by names within the root folder and its subdirectories.
    
    Parameters:
    root_folder (str): The path to the root folder.
    target_folder_names (list of str): The list of folder names to search for.
    
    Returns:
    dict: A dictionary where the keys are the folder names and the values are their paths. 
          If a folder is not found, its value will be None.
    """
    for key ,value in root_folder.items(): 
        for dirpath, dirnames, filenames in os.walk(value['path']):
            for file in filenames:
                if file in target_file_names:
                    file_data ={
                        'file_name': file,
                        'folder_path': dirpath
                    }
                    return file_data





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
    parent_folder ='vectorstore/loan_db_faiss/loan_search/loan_floder'
    folder_list = get_folder_list(parent_folder)
    folder_name = get_folder_names_from_gpt(query,folder_list)
    logger.info(f'Folder Name Get from Gpt: {folder_name}')
    folder_path = find_folder_by_name(parent_folder,folder_name)
    logger.info(f'folder path: {folder_path}')
    file_list = get_specific_folder_list(folder_path)
    file_name = get_file_names_from_gpt(query,file_list)
    logger.info(f'file name get from gpt: {file_name}')
    vector_database_path = 'vectorstore/loan_db_faiss'
    index_file =[]
    for file in file_name:
        file_path = find_file_by_name(folder_path,[file])
        complete_path_db = file_path['folder_path']
        if os.path.exists(complete_path_db):
            logger.info(f'Folder path in database Exits at: {complete_path_db}')
            base_file_name = os.path.splitext(file_path['file_name'])[0]
            pkl_path = os.path.join(complete_path_db,base_file_name+'.pkl')
            faiss_path = os.path.join(complete_path_db,base_file_name+'.faiss')
            if os.path.exists(pkl_path) and os.path.exists(faiss_path):
                logger.info(f'Succesfully Found index of File "{pkl_path} \n Index of file {os.path.splitext(file)[0]}"')
                index_file.append({"index":os.path.splitext(file)[0],"folder_path":file_path['folder_path']})
        else:
            logger.error(f'Folder in database not foundat : {complete_path_db}')
    return index_file

def question_answer():
    try:
        question = f"Tell me daily gross earningof Rahul?"
        db_index = get_vector_db_index(question)
        answer_list = []
        for index in db_index:
            vectorstore = load_vectorstore(folder_path=index['folder_path'],index=index['index'])
            answer = get_answer(question, vectorstore)
            # Extract the result from the answer
            answer_list.append(answer.get('result', 'No result found.'))
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=350,  # Adjust the max_tokens as needed
            timeout=None,
            max_retries=2
        )
        prompt = (
            f"Given the following question, extract relevant information from the provided file data and generate a new answer based on the question. "
            f"Ensure that you analyze the content of the files to provide the most accurate and relevant response. "
            f"Respond with a summary that combines information from all relevant files."
            f"\n\nQuestion: {question}\n\nList of file data: {answer_list}\n"
            )
        result = llm.invoke(prompt)
        
        # Print the formatted result
        clear_terminal()
        print("\n--- Payslip Information ---")
        print(result.content)
        print("\n-----------------------------")
    except FileNotFoundError as e:
        logger.error(e)

question_answer()

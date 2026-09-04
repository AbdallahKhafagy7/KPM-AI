import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def preprocess_file(file_path: str):
    """
    Loads a PDF file, splits it into document chunks, and deletes the local file.
    
    Returns:
        list[Document]: The list of generated LangChain document chunks.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File at {file_path} does not exist.")

    # 1. Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Clean up the processed file
    os.remove(file_path)

    return chunks
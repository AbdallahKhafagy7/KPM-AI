from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


#load database
vectorstore = Chroma(
    collection_name="data-context",
    embedding_function=embeddings,
    persist_directory="./db"
)


def add_docs(chunks):
    """
    Add documents to the vector store.

    Args:
        chunks (list): List of document chunks to be added.

    """
    vectorstore.add_documents(chunks, embedding=embeddings)
    
    return {'added': True,}
    
    
def update_docs(chunks):
    """
    Update documents in the vector store.

    Args:
        chunks (list): List of document chunks to be updated.

    """
    ids = [doc.id for doc in chunks if doc.id is not None]
    if ids:
        vectorstore.update_documents(ids=ids, documents=chunks)
        
    return {'updated': True}
    
def delete_docs(chunks):
    """
    Delete documents in the vector store.

    Args:
        chunks (list): List of document chunks to be deleted.

    """
    
    ids = [str(doc.id) for doc in chunks if doc.id is not None]
    if ids:
        vectorstore.delete(ids=ids)
    return {'deleted': True}

def similarity_search(query: str, k: int = 4):
    """Retrieve relevant chunks from the database."""
    results = vectorstore.similarity_search(query, k=k)
    return [
        {"content": doc.page_content, "metadata": doc.metadata, "id": doc.id}
        for doc in results
    ]
     



from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ollama langchain model
llm = OllamaLLM(model="qwen2.5:0.5b")

# vector store
vectorstore = Chroma(
    collection_name="data-context",
    embedding_function=embeddings,
    persist_directory="./db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


def getResponse(query: str):
    # qa chain prompt template ({context} + {input})
    qa_prompt_template = """Use the following pieces of context to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {input}

Answer:"""

    # prompt template (qa chain prompt)
    qa_prompt = PromptTemplate(
        template=qa_prompt_template, 
        input_variables=["context", "input"]
    )

    # prompt template for retrieved document chunks format ({page_content} + metadata)
    document_prompt = PromptTemplate(
        template="Content: {page_content}",
        input_variables=["page_content"]
    )

    # stuff document chain (llm + prompt + formatted document template)
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt,
        document_prompt=document_prompt
    )

    # retrievalQA (combine_docs_chain + retriever)
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )

    # res = final_chain(query)
    response = retrieval_chain.invoke({"input": query})

    # return res (response + source_documents)
    return {
        "answer": response.get("answer"),
        "source_documents": response.get("context")
    }
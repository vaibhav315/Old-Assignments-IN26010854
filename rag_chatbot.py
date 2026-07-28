# ===============================================
# RAG CHATBOT UTILITIES
# ===============================================


import os


from langchain_community.document_loaders import PyPDFLoader


from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain_openai import OpenAIEmbeddings


from langchain_community.vectorstores import FAISS





def process_pdf(file_path):


    loader = PyPDFLoader(
        file_path
    )


    documents = loader.load()



    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200

    )


    chunks = splitter.split_documents(

        documents

    )


    return chunks





def create_vector_database(chunks):


    embeddings = OpenAIEmbeddings()



    vector_database = FAISS.from_documents(

        chunks,

        embeddings

    )


    vector_database.save_local(

        "vectorstore"

    )


    return vector_database





def load_vector_database():


    embeddings = OpenAIEmbeddings()



    database = FAISS.load_local(

        "vectorstore",

        embeddings,

        allow_dangerous_deserialization=True

    )


    return database

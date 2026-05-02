from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

def load_pdf_files(data):
    loader=DirectoryLoader(
        data, 
        glob="*.pdf", 
        loader_cls=PyPDFLoader

    )
    documents=loader.load()
    return documents


extracted_data = load_pdf_files("data/")
def filter_to_minimal_docs(docs:List[Document])->List[Document]:
    """Give a list of Docuemnts objects, return a new list of Docuement object
    containing only 'source' in metadata and the original page_content"""
    minimal_docs:List[Document]=[]
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return minimal_docs
minimal_docs=filter_to_minimal_docs(extracted_data)

#split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
    )
    texts_chunks=text_splitter.split_documents(minimal_docs)
    return texts_chunks



def download_hugging_face_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


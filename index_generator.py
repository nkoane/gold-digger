import pypdf
import os
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader

import chromadb

# load environment variables
load_dotenv()

storage_dir = "./storage/nemisa"
documents_dir = "./data/nemisa"
# read the files;

embeddings = OpenAIEmbeddings()

client_settings = chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=storage_dir,
    anonymized_telemetry=False
)

print('Creating new index')

# how do i read files from the local directory?

files = os.listdir(documents_dir)
loaders = []
docs = []


for file in files:
    loader = PyPDFLoader(documents_dir + '/' + file)
    # pages = loader.load_and_split()
    docs.extend(loader.load_and_split())
    print('loaded file:', file, len(docs))
    # break

text_splitter = text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0)
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

vectordb = Chroma(
    collection_name="nemisa_annaul_reports",
    persist_directory=storage_dir,
    client_settings=client_settings,
    embedding_function=embeddings
)

# vectordb.delete_collection()
vectordb.add_documents(documents=documents, embedding=embeddings)
vectordb.persist()

print(files, len(docs), len(documents))

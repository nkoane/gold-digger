from os import path
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, LLMPredictor, ServiceContext
from dotenv import load_dotenv
import streamlit as st
from langchain import OpenAI


import logging
import sys
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

st.title("Llama 🚀")

storage_dir = "./storage"

llm_predictor = LLMPredictor(llm=OpenAI(
    temperature=0, model_name="text-davinci-003"))

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# check if file exists
if path.exists('storage') & path.exists('storage/docstore.json') & path.exists('storage/index_store.json') & path.exists('storage/vector_store.json'):
    st.write('Loading from storage')
    storage_context = StorageContext.from_defaults(
        persist_dir=storage_dir)
    index = load_index_from_storage(
        storage_context, service_context=service_context)
else:
    st.write('Creating new index')
    documents = SimpleDirectoryReader('data').load_data()
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context)
    index.storage_context.persist(persist_dir=storage_dir)

# print(index)
query_engine = index.as_query_engine()

query = st.text_input("Enter your question below:",
                      "What is the meaning of life?")

response = query_engine.query(query)
st.write(response)

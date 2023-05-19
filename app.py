from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferMemory
import streamlit as st
from os import path
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks.base import BaseCallbackHandler

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain

import chromadb

# load environment variables
load_dotenv()

st.set_page_config(
    page_title="Gold Diggin'",
    layout="wide",
)

storage_dir = "./storage/nemisa"
documents_dir = "./data/nemisa"
# read the files;


class SessionState:
    def __init__(self):
        self.data = []


state = SessionState()


def append_data(data):
    state.data.append(data)


global output

print(len)


class MyStreamingCallBackHandler(BaseCallbackHandler):

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs) -> None:
        output.write("...")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        append_data(token)
        # convert a list to a string
        str = ''.join(state.data)
        output.markdown(str)
        # print(token, end="", flush=True)


embeddings = OpenAIEmbeddings()

client_settings = chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=storage_dir,
    anonymized_telemetry=False
)

print('Loading from storage')
vectorstore = Chroma(
    collection_name="nemisa_annaul_reports",
    persist_directory=storage_dir,
    client_settings=client_settings,
    embedding_function=embeddings
)

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=1), vectorstore.as_retriever(), memory=memory, verbose=True, callbacks=[MyStreamingCallBackHandler()])

st.title("Golden, ⛏️")
question = st.text_input(
    "Ask a question")

output = st.empty()
vectordbkwargs = {"search_distance": 0.9}
if question:
    response = result = qa({"question": question}, vectordbkwargs)
    output.markdown(response['answer'])

import streamlit as st
from os import path
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks.base import BaseCallbackHandler

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFDirectoryLoader
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

embeddings = OpenAIEmbeddings()

client_settings = chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=storage_dir,
    anonymized_telemetry=False
)

if path.exists(storage_dir + '/index'):
    print('Loading from storage')
    vectordb = Chroma(
        collection_name="nemisa_annaul_reports",
        persist_directory=storage_dir,
        client_settings=client_settings,
        embedding_function=embeddings
    )
else:
    print('Creating new index')
    loader = PyPDFDirectoryLoader(documents_dir)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    vectordb = Chroma(
        collection_name="nemisa_annaul_reports",
        persist_directory=storage_dir,
        client_settings=client_settings,
        embedding_function=embeddings
    )
    vectordb.add_documents(documents=texts, embedding=embeddings)
    vectordb.persist()


class SessionState:
    def __init__(self):
        self.data = []


state = SessionState()


def append_data(data):
    state.data.append(data)


global output


class MyStreamingCallBackHandler(BaseCallbackHandler):

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs) -> None:
        output.text("Thinking ...")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        append_data(token)
        # convert a list to a string
        str = ''.join(state.data)
        output.markdown(str)
        # print(token, end="", flush=True)


chat_model = ChatOpenAI(streaming=True, callbacks=[MyStreamingCallBackHandler()],
                        temperature=0, model_name="gpt-3.5-turbo")


questions = []
chat_history = []

qAndy = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=vectordb.as_retriever(),
    verbose=True,
    # qa_prompt=QA_PROMPT,
    # condense_question_prompt=CONDENSE_QUESTION_PROMPT
)

st.title("Golden, ⛏️")
question = st.text_input(
    "Ask a question")

output = st.empty()

if question:
    result = qAndy({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))

exit()

"""
if question:
    questions.append(question)
    chat_history.append((question, result['answer']))
    print(result)
    # mm = chain.run(text=query)

"""


# Initialize session state
st.title("Golden, ⛏️")


chat = ChatOpenAI(streaming=True, callbacks=[MyStreamingCallBackHandler()],
                  temperature=0.4, model_name="gpt-3.5-turbo")

template = "You are a helpful assistant that only answers the given question in markdown format"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt)

# qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type='stuff', vectorstore=vectordb)
qa = RetrievalQAWithSourcesChain.from_chain_type(
    llm=OpenAI(), chain_type='stuff', retriever=vectordb.as_retriever())

query = st.text_input(
    "Ask a question")

output = st.empty()


if query:
    results = qa(query)
    print(results)
    # mm = chain.run(text=query)

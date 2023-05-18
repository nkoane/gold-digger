import streamlit as st
from os import path
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()

storage_dir = "./storage/nemisa"

st.set_page_config(
    page_title="Gold Diggin'",
    layout="wide",
)

# Initialize session state


st.title("Golden, ⛏️")


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


chat = ChatOpenAI(streaming=True, callbacks=[MyStreamingCallBackHandler()],
                  temperature=0.4, model_name="gpt-3.5-turbo")

template = "You are a helpful assistant that only answers the given question in markdown format"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt)

query = st.text_input(
    "Ask a question")

output = st.empty()

if query:
    mm = chain.run(text=query)

# print(result)
# st.write(mm)

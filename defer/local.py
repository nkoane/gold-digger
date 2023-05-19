from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st

local_path = './models/ggml-gpt4all-j-v1.3-groovy.bin'

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]
# Verbose is required to pass to the callback manager
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)
# If you want to use a custom model add the backend parameter
# Check https://docs.gpt4all.io/gpt4all_python.html for supported backends
llm = GPT4All(model=local_path, backend='gptj',
              callbacks=callbacks, verbose=True)

st.title("Local is lekker, 🤯")

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who is the president of the United States?"

response = llm_chain.run(question)

question = st.text_input("Question", value=question)
st.write(response)

print(response)

import streamlit as st
import pandas as pd
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatOllama(
    model="llama3",
    max_tokens=100,
)

messages = []
st.title("Chatbot with History")
st.write("Welcome to my chatbot")

prompt = st.chat_input("Enter your message: ")

if prompt:
    messages.append(HumanMessage(content=prompt))
    response = llm.invoke(messages)
    messages.append(AIMessage(content=response.content))

    st.write("Bot: " + response.content)
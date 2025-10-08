from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage , AIMessage

llm = ChatOllama(
    model="llama3",
    max_tokens=100
)

while True:
    prompt = input("User: ")
    if prompt.lower() == "exit":
        break

    response = llm.invoke(prompt)
    print("Bot:", response.content)

print("Chat ended.")
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatOllama(
    model="llama3",
    max_tokens=100,
    temperature=0.7,
    n=2
)

messages = []

while True:
    prompt = input("User: ")
    if prompt.lower() == "exit":
        break

    messages.append(HumanMessage(content=prompt))
    response = llm.invoke(messages)
    messages.append(AIMessage(content=response.content))

    print("Bot:", response.content)

print("Chat ended.")
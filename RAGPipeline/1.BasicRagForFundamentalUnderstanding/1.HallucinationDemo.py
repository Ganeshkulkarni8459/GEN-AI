from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3",
    max_tokens=100,
)

prompt = "What is Vaibhav Zodge?"

response = model.invoke(prompt)

print(response.content)
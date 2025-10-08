from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="llama3",
    max_tokens=100
)

prompt = "What is the capital of France?"
response = llm.invoke(prompt)
print(response)

prompt = "Previous Response: " + response + "What is good in it?"
response = llm.invoke(prompt)
print(response)

from flask import Flask, request, jsonify, render_template
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatOllama(
    model="llama3",
    max_tokens=100,
)

messages = []

app = Flask(__name__)

def root():
    return render_template('index.html')

@app.route('/',methods=['GET'])
def index():
    return root()

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')

    print("******** User message : "+user_message)
    return jsonify({'response': get_response(user_message)})

def get_response(prompt):
    messages.append(HumanMessage(content=prompt))
    response = llm.invoke(messages)
    messages.append(AIMessage(content=response.content))
    return response.content

app.run("0.0.0.0", port=5400, debug=True)

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="Llama3.2")

def detect_action_with_model(user_input: str):
    """
    Use the model itself to decide whether the task is TEXT, IMAGE, AUDIO, or TRANSLATION.
    """
    classification_prompt = f"""
    You are an intelligent assistant. 
    Based on the following user input, decide what the user wants to do.
    You must answer with only one of these categories (no extra text):

    - IMAGE → if the user wants to generate, draw, or create an image.
    - AUDIO → if the user wants speech, sound, or convert text to audio.
    - TRANSLATION → if the user wants to translate text from one language to another.
    - TEXT → for all other general text-based queries.

    User input: "{user_input}"
    Respond with exactly one of: IMAGE, AUDIO, TRANSLATION, TEXT.
    """

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    action = response.content.strip().upper()

    return action

def main():
    user_input = input("Enter your prompt here: ")

    action_type = detect_action_with_model(user_input)
    print("Detected Action:", action_type)

if __name__ == "__main__":
    main()

from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_ollama import ChatOllama
from openai import OpenAI
import re, os
import requests
from gtts import gTTS

app = Flask(__name__)
CORS(app)

# ---- Initialize models ---- #
ollama = ChatOllama(model="llama3", max_tokens=300)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Example Image model API (Replicate Stable Diffusion or any local endpoint)
STABLE_DIFFUSION_API = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


# ---- Action Detection ---- #
def detect_action(prompt: str):
    norm = prompt.lower()
    if any(k in norm for k in ["image", "draw", "photo", "picture", "illustrate"]):
        return "IMAGE"
    elif any(k in norm for k in ["audio", "text to speech", "tts", "convert to audio", "speak"]):
        return "AUDIO"
    elif "translate" in norm or "translation" in norm or re.search(r"from .* to .*", norm):
        return "TRANSLATION"
    else:
        return "TEXT"


# ---- Prompt Enrichment ---- #
def augment_prompt(prompt: str, action: str):
    if action == "IMAGE":
        return f"Generate a high-quality, detailed image for: {prompt}. Realistic, high-resolution, vibrant colors."
    elif action == "AUDIO":
        return f"Convert this text into a natural, human-like voice in MP3 format: {prompt}"
    elif action == "TRANSLATION":
        return f"Translate accurately while preserving tone and meaning: {prompt}"
    else:
        return f"Provide a clear, concise, and structured response to: {prompt}"


# ---- Generators ---- #

def generate_text(prompt: str):
    resp = ollama.invoke([{"role": "user", "content": prompt}])
    return resp.content

def generate_translation(prompt: str):
    resp = ollama.invoke([{"role": "user", "content": prompt}])
    return resp.content

def generate_audio(prompt: str):
    os.makedirs("static", exist_ok=True)
    filename = "static/generated_audio.mp3"
    tts = gTTS(prompt)
    tts.save(filename)
    return filename

def generate_image(prompt: str):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    response = requests.post(STABLE_DIFFUSION_API, headers=headers, json={"inputs": prompt})
    if response.status_code == 200:
        # Save as image file
        os.makedirs("static", exist_ok=True)
        img_path = "static/generated_image.png"
        with open(img_path, "wb") as f:
            f.write(response.content)
        return img_path
    else:
        return f"Image generation failed: {response.text}"


# ---- API Route ---- #
@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Step 1: Detect action
    action = detect_action(prompt)

    # Step 2: Augment prompt
    augmented_prompt = augment_prompt(prompt, action)

    try:
        # Step 3: Route to model
        if action == "IMAGE":
            result = generate_image(augmented_prompt)
        elif action == "AUDIO":
            result = generate_audio(augmented_prompt)
        elif action == "TRANSLATION":
            result = generate_translation(augmented_prompt)
        else:
            result = generate_text(augmented_prompt)

        return jsonify({
            "action": action,
            "augmented_prompt": augmented_prompt,
            "result": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5400, debug=True)

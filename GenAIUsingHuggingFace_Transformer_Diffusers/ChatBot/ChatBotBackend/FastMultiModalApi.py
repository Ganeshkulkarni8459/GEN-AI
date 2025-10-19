from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
import os
import uuid
import requests
from pathlib import Path
import base64

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create directories
Path("static/images").mkdir(parents=True, exist_ok=True)
Path("static/audio").mkdir(parents=True, exist_ok=True)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/process")
async def process_prompt(request: PromptRequest):
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(400, "Prompt cannot be empty")
    
    try:
        # Detect intent
        intent = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Classify as: IMAGE, AUDIO, TRANSLATION, or TEXT"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=10
        )
        
        intent_type = intent.choices[0].message.content.strip().upper()
        
        # Route to handler
        if intent_type == "IMAGE":
            return handle_image(prompt)
        elif intent_type == "AUDIO":
            return handle_audio(prompt)
        elif intent_type == "TRANSLATION":
            return handle_translation(prompt)
        else:
            return handle_text(prompt)
            
    except Exception as e:
        raise HTTPException(500, str(e))

def handle_text(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return {
        "intent": "Text Generation",
        "type": "text",
        "result": response.choices[0].message.content.strip()
    }

def handle_image(prompt):
    # Generate image
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024"
    )
    
    # Download image
    image_url = response.data[0].url
    image_bytes = requests.get(image_url).content
    
    # Save file
    filename = f"generated_{uuid.uuid4().hex}.png"
    filepath = f"static/images/{filename}"
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    
    # Convert to base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    return {
        "intent": "Image Generation",
        "type": "image",
        "result": "Image generated successfully",
        "file_path": filepath,
        "image_data": f"data:image/png;base64,{image_base64}"
    }

def handle_translation(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Translate English to Marathi. Return only the translation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    
    marathi_text = response.choices[0].message.content.strip()
    
    # Extract original text
    original = prompt
    if ":" in prompt and "translate" in prompt.lower():
        original = prompt.split(":", 1)[1].strip()
    
    return {
        "intent": "Translation (English → Marathi)",
        "type": "translation",
        "result": marathi_text,
        "original": original
    }

def handle_audio(prompt):
    # Extract text to speak
    text = prompt
    if ":" in prompt:
        text = prompt.split(":", 1)[1].strip()
    
    # Generate audio
    filename = f"audio_{uuid.uuid4().hex}.mp3"
    filepath = f"static/audio/{filename}"
    
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    
    response.stream_to_file(filepath)
    
    # Convert to base64
    with open(filepath, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    return {
        "intent": "Text-to-Speech",
        "type": "audio",
        "result": f"Audio generated for: {text[:100]}",
        "file_path": filepath,
        "audio_data": f"data:audio/mpeg;base64,{audio_base64}"
    }

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return {"message": "AI Multi-Modal API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "api_key_set": bool(os.getenv("OPENAI_API_KEY"))}

if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
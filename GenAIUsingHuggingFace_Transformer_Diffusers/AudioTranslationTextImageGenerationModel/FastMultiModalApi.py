from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
import os
import uuid
import requests
from pathlib import Path
import base64
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create necessary directories
Path("static/images").mkdir(parents=True, exist_ok=True)
Path("static/audio").mkdir(parents=True, exist_ok=True)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/process")
async def process_prompt(request: PromptRequest):
    user_input = request.prompt.strip()
    
    if not user_input:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    # Detect intent using GPT
    try:
        intent_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an intent classifier. Analyze the user's request and respond with EXACTLY ONE of these words:
- IMAGE: if they want to generate, create, draw, or visualize an image
- AUDIO: if they want text-to-speech, audio generation, or to hear something read aloud
- TRANSLATION: if they want to translate text to Marathi or convert English to Marathi
- TEXT: for general questions, conversations, explanations, or text generation

Respond with only the category word, nothing else."""
                },
                {"role": "user", "content": user_input}
            ],
            temperature=0.3,
            max_tokens=10
        )
        
        action_type = intent_response.choices[0].message.content.strip().upper()
        
        # Route to appropriate handler
        if action_type == "IMAGE":
            return await handle_image(user_input)
        elif action_type == "AUDIO":
            return await handle_audio(user_input)
        elif action_type == "TRANSLATION":
            return await handle_translation(user_input)
        elif action_type == "TEXT":
            return await handle_text(user_input)
        else:
            return await handle_text(user_input)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

async def handle_text(prompt: str):
    """Generate text response using GPT"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Provide clear, concise, and informative responses."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        result = response.choices[0].message.content.strip()
        
        return {
            "intent": "Text Generation",
            "type": "text",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

async def handle_image(prompt: str):
    """Generate image using DALL-E"""
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        
        image_url = response.data[0].url
        
        # Download and save image
        image_data = requests.get(image_url)
        filename = f"generated_{uuid.uuid4().hex}.png"
        filepath = f"static/images/{filename}"
        
        with open(filepath, "wb") as f:
            f.write(image_data.content)
        
        return {
            "intent": "Image Generation",
            "type": "image",
            "result": "Image generated successfully",
            "file_path": filepath
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

async def handle_translation(prompt: str):
    """Translate English to Marathi using GPT"""
    try:
        # Extract text to translate
        translation_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a professional English to Marathi translator. 
Translate the given English text to Marathi accurately. 
If the input contains phrases like "translate to Marathi:" or "convert to Marathi:", extract only the actual text to translate.
Provide only the Marathi translation, nothing else."""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        marathi_text = translation_response.choices[0].message.content.strip()
        
        # Extract original text
        original_text = prompt
        if "translate" in prompt.lower():
            parts = prompt.split(":", 1)
            if len(parts) > 1:
                original_text = parts[1].strip()
        
        return {
            "intent": "Translation (English → Marathi)",
            "type": "translation",
            "result": marathi_text,
            "original": original_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

async def handle_audio(prompt: str):
    """Generate audio using OpenAI TTS"""
    try:
        # Extract text to speak
        text_to_speak = prompt
        if "read" in prompt.lower() or "speak" in prompt.lower():
            parts = prompt.split(":", 1)
            if len(parts) > 1:
                text_to_speak = parts[1].strip()
            else:
                # Try to extract after common phrases
                for phrase in ["read this", "read aloud", "say this", "speak this"]:
                    if phrase in prompt.lower():
                        text_to_speak = prompt.lower().split(phrase, 1)[1].strip()
                        break
        
        filename = f"audio_{uuid.uuid4().hex}.mp3"
        filepath = f"static/audio/{filename}"
        
        # Generate speech
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
            input=text_to_speak
        )
        
        # Save audio file
        response.stream_to_file(filepath)
        
        return {
            "intent": "Text-to-Speech",
            "type": "audio",
            "result": f"Audio generated for: {text_to_speak[:100]}{'...' if len(text_to_speak) > 100 else ''}",
            "file_path": filepath
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

# Mount static files - this serves images and audio
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {"message": "AI Multi-Modal Service API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "openai_key_set": bool(os.getenv("OPENAI_API_KEY"))}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
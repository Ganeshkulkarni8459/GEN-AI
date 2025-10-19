"""
AI Multi-Modal API Backend
FastAPI server with OpenAI integration for text, image, audio generation and translation
"""

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
from typing import Dict, Any

# Initialize FastAPI app
app = FastAPI(
    title="AI Multi-Modal API",
    description="Generate text, images, audio, and translations using OpenAI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("WARNING: OPENAI_API_KEY environment variable not set!")
    
client = OpenAI(api_key=openai_api_key)

# Create necessary directories
IMAGES_DIR = Path("static/images")
AUDIO_DIR = Path("static/audio")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Pydantic models
class PromptRequest(BaseModel):
    prompt: str

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Create an image of a sunset over mountains"
            }
        }

# API Routes
@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint - API health check"""
    return {
        "message": "AI Multi-Modal Service API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint with system status"""
    return {
        "status": "healthy",
        "openai_key_set": bool(openai_api_key),
        "static_dirs_exist": {
            "images": IMAGES_DIR.exists(),
            "audio": AUDIO_DIR.exists()
        },
        "images_count": len(list(IMAGES_DIR.glob("*.png"))) if IMAGES_DIR.exists() else 0,
        "audio_count": len(list(AUDIO_DIR.glob("*.mp3"))) if AUDIO_DIR.exists() else 0
    }

@app.post("/process")
async def process_prompt(request: PromptRequest) -> Dict[str, Any]:
    """
    Main endpoint to process user prompts
    Detects intent and routes to appropriate handler
    """
    user_input = request.prompt.strip()
    
    if not user_input:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    if not openai_api_key:
        raise HTTPException(
            status_code=500, 
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
        )
    
    try:
        # Detect intent using GPT
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
        print(f"[INFO] Detected intent: {action_type} for prompt: '{user_input[:50]}...'")
        
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
            # Default to text if intent is unclear
            print(f"[WARNING] Unknown intent: {action_type}, defaulting to TEXT")
            return await handle_text(user_input)
            
    except Exception as e:
        print(f"[ERROR] Error in process_prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Handler Functions
async def handle_text(prompt: str) -> Dict[str, Any]:
    """Generate text response using GPT"""
    try:
        print(f"[INFO] Generating text response...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Provide clear, concise, and informative responses."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        result = response.choices[0].message.content.strip()
        print(f"[SUCCESS] Text generated: {len(result)} characters")
        
        return {
            "intent": "Text Generation",
            "type": "text",
            "result": result
        }
    except Exception as e:
        print(f"[ERROR] Error in handle_text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

async def handle_image(prompt: str) -> Dict[str, Any]:
    """Generate image using DALL-E and return as base64"""
    try:
        print(f"[INFO] Generating image with DALL-E...")
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        
        image_url = response.data[0].url
        print(f"[INFO] Image generated, downloading from: {image_url}")
        
        # Download image
        image_response = requests.get(image_url, timeout=30)
        image_response.raise_for_status()
        image_bytes = image_response.content
        
        # Save to file (for backup)
        filename = f"generated_{uuid.uuid4().hex}.png"
        filepath = IMAGES_DIR / filename
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        print(f"[SUCCESS] Image saved to: {filepath}")
        
        # Convert to base64 for frontend display
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        print(f"[INFO] Image converted to base64 (size: {len(image_base64)} chars)")
        
        return {
            "intent": "Image Generation",
            "type": "image",
            "result": "Image generated successfully",
            "file_path": str(filepath),
            "image_data": f"data:image/png;base64,{image_base64}"
        }
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network error in handle_image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        print(f"[ERROR] Error in handle_image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

async def handle_translation(prompt: str) -> Dict[str, Any]:
    """Translate English to Marathi using GPT"""
    try:
        print(f"[INFO] Translating to Marathi...")
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
        
        print(f"[SUCCESS] Translation completed")
        
        return {
            "intent": "Translation (English → Marathi)",
            "type": "translation",
            "result": marathi_text,
            "original": original_text
        }
    except Exception as e:
        print(f"[ERROR] Error in handle_translation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

async def handle_audio(prompt: str) -> Dict[str, Any]:
    """Generate audio using OpenAI TTS and return as base64"""
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
        
        print(f"[INFO] Generating audio for: '{text_to_speak[:50]}...'")
        filename = f"audio_{uuid.uuid4().hex}.mp3"
        filepath = AUDIO_DIR / filename
        
        # Generate speech
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
            input=text_to_speak
        )
        
        # Save audio file
        response.stream_to_file(str(filepath))
        print(f"[SUCCESS] Audio saved to: {filepath}")
        
        # Read file and convert to base64 for frontend display
        with open(filepath, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        print(f"[INFO] Audio converted to base64 (size: {len(audio_base64)} chars)")
        
        return {
            "intent": "Text-to-Speech",
            "type": "audio",
            "result": f"Audio generated for: {text_to_speak[:100]}{'...' if len(text_to_speak) > 100 else ''}",
            "file_path": str(filepath),
            "audio_data": f"data:audio/mpeg;base64,{audio_base64}"
        }
        
    except Exception as e:
        print(f"[ERROR] Error in handle_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

# Mount static files for backup access
app.mount("/static", StaticFiles(directory="static"), name="static")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("=" * 60)
    print("AI Multi-Modal Service Starting...")
    print("=" * 60)
    print(f"OpenAI API Key: {'✓ Set' if openai_api_key else '✗ Not Set'}")
    print(f"Images Directory: {IMAGES_DIR.absolute()}")
    print(f"Audio Directory: {AUDIO_DIR.absolute()}")
    print("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print("\n" + "=" * 60)
    print("AI Multi-Modal Service Shutting Down...")
    print("=" * 60)

# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    HOST = "0.0.0.0"
    PORT = 8000
    
    print("\n" + "=" * 60)
    print(f"Starting FastAPI server on http://{HOST}:{PORT}")
    print("=" * 60)
    print(f"OpenAI API Key: {'✓ Configured' if openai_api_key else '✗ NOT CONFIGURED'}")
    print("=" * 60)
    
    if not openai_api_key:
        print("\n⚠️  WARNING: OpenAI API key not found!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'\n")
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )
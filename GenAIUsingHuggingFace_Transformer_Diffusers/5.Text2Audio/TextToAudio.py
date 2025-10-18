import pyttsx3

# Initialize the TTS engine
engine = pyttsx3.init()

# Configure voice properties
engine.setProperty('rate', 150)    # Speed of speech (default is 200)
engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

# Get available voices
voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[0].id)  # Male voice (usually)
# engine.setProperty('voice', voices[1].id)  # Female voice (usually)

# Text to convert
text = "Hello! This is offline text to speech. It works without internet connection."

# Speak the text
engine.say(text)
engine.runAndWait()

# Save to file
engine.save_to_file(text, 'output_audio.mp3')
engine.runAndWait()

print("Audio saved as 'output_audio.mp3'")
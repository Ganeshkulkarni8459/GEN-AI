from transformers import pipeline, logging

logging.set_verbosity_error()

pipeline = pipeline(model="openai/whisper-medium")

audio_file = "real.mp3"

text = pipeline(audio_file)

print(text['text'])
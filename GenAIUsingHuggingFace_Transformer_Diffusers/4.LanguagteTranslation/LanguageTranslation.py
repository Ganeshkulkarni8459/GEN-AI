from transformers import pipeline,logging
import tkinter as tk
from tkinter import messagebox

logging.set_verbosity_error()

# Step 1: Define model for Marathi

MODELS = {
    'marathi': 'Helsinki-NLP/opus-mt-en-mr'
}

translator = pipeline("translation_en_to_marathi",model=MODELS['marathi'],device='cpu')

# Step 2: Translate sentences
texts = "I Love programming in python"
translation = translator(texts,max_length=100)
print("English:",texts)
print("Marathi: ",translation[0]['translation_text'])
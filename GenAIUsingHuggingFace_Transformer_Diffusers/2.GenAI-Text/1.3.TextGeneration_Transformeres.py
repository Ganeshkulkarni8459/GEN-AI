from transformers import pipeline , logging

logging.set_verbosity_error()

classifier = pipeline(
    "text-generation",
    model="openai-community/gpt2",
    device="cpu"
)


result = classifier("Oh great, another Monday morning meeting at 7am")
print(result[0]['generated_text'])
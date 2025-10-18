from transformers import pipeline , logging

logging.set_verbosity_error()

classifier = pipeline("text-classification", device="cpu")

result = classifier("Oh great, another Monday morning meeting at 7am")
print(result)
result = classifier("The customer support was so helpful — they put me on hold for an hour.")
print(result)
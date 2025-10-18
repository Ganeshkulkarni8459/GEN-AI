from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt = "Generate logo for my company which provides education company name : Dnyanyog Education"

response = client.images.generate(
    model="dall-e-3",
    prompt=prompt,
    size="1024x1024",
    quality="standard",
    n=1,
)

image_url = response.data[0].url

# Download and save
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
image.save("astronaut_horse.png")
image.show()
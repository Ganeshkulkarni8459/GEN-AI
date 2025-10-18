import os
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# For image variation (similar style)
with open("astronaut_horse.png", "rb") as image_file:
    response = client.images.create_variation(
        image=image_file,
        n=1,
        size="1024x1024"
    )

image_url = response.data[0].url

# Download and save
response = requests.get(image_url)
output_image = Image.open(BytesIO(response.content))
output_image.save("image2image.png")
print("Image saved as 'image2image.png'")
output_image.show()
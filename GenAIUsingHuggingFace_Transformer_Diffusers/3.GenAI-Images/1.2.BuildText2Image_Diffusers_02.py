from diffusers import AutoPipelineForText2Image
import torch

pipe_txt2img = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    torch_dtype=torch.float32
).to("cpu")

prompt = "cinematic photo of Godzilla eating sushi with a cat in a izakaya"
image = pipe_txt2img(prompt, num_inference_steps=20).images[0]
image.save("generated_image.png")
image.show()
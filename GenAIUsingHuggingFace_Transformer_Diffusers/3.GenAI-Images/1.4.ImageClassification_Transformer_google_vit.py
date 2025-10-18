from transformers import pipeline

# Create the image classification pipeline
classifier = pipeline(task="image-classification", model="google/vit-base-patch16-224")

# Run classification on the image
result = classifier("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")

# Print the results
print(result)
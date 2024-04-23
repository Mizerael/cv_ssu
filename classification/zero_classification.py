from transformers import pipeline
from PIL import Image
from tqdm import tqdm
import os
import json

images_dir = "images"
find_dir = "find"

candidate = ["cases", "human", "weapon", "sticers", "music"]

names = []

predictions = []

for file in os.listdir(images_dir):
    names.append(file)


checkpoint = "openai/clip-vit-large-patch14"
detector = pipeline(
    model=checkpoint, task="zero-shot-image-classification", cache_dir="./cache"
)


for file in tqdm(names):
    image = Image.open(os.path.join(images_dir, file))
    prediction = detector(image, candidate_labels=candidate)
    predictions.append({"image": file, "prediction": prediction})

with open("predictions.json", "w") as f:
    json.dump(predictions, f)

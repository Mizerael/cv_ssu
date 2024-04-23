from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import os
import gc
from tqdm import tqdm

images_dir = "images"
find_dir = "find"

count = 10

batchSize = 2

random_search_text = ["cases", "human", "weapon"]

if not os.path.exists(find_dir):
    os.makedirs(find_dir)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir="./cache")
processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch16", cache_dir="./cache"
)

result = []
names = []
for file in os.listdir(images_dir):
    names.append(file)

for i in tqdm(range(len(names) // batchSize + (len(names) % batchSize > 0))):
    batch_names = names[i * batchSize : (i + 1) * batchSize]

    batch_images = []
    for file in batch_names:
        image = Image.open(os.path.join(images_dir, file))
        batch_images.append(image)

    inputs = processor(
        text=random_search_text, images=batch_images, return_tensors="pt"
    )
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    for j in range(len(batch_names)):
        result.append((names[i * batchSize + j], probs[j][2]))

    for image in batch_images:
        image.close()
        del image
    gc.collect()
    torch.cuda.empty_cache()


result.sort(key=lambda x: x[1], reverse=True)

for i in range(count):
    image = Image.open(os.path.join(images_dir, result[i][0]))
    image.save(os.path.join(find_dir, result[i][0]), "PNG")
    image.close()
    del image

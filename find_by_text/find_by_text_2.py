from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
import os
from tqdm import tqdm

images_dir = "images"
find_dir = "find2"

count = 10

batchSize = 2

random_search_text = ["weapon"]

if not os.path.exists(find_dir):
    os.makedirs(find_dir)


names = []

for file in os.listdir(images_dir):
    names.append(file)


checkpoint = "google/owlv2-base-patch16-ensemble"
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    checkpoint, cache_dir="./cache"
)

processor = AutoProcessor.from_pretrained(checkpoint)

for file in tqdm(names):
    image = Image.open(os.path.join(images_dir, file)).convert('RGB')
    inputs = processor(text=random_search_text, images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, threshold=0.1, target_sizes=target_sizes
        )[0]

    draw = ImageDraw.Draw(image)

    scores = results["scores"].tolist()
    labels = results["labels"].tolist()
    boxes = results["boxes"].tolist()

    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
        draw.text(
            (xmin, ymin), f"{random_search_text}: {round(score,2)}", fill="white"
        )
        if xmin != xmax and ymin != ymax:
            image.save(os.path.join(find_dir, file), 'PNG')
    image.close()



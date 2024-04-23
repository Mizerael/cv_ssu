import os
from os import listdir
from os.path import isfile, join
from PIL import Image
from imagehash import average_hash
from tqdm import tqdm

images_path = 'images'
unique_images_path = 'unique_images'
if not os.path.exists(unique_images_path):
    os.makedirs(unique_images_path)

unique_images = set()

for file in tqdm(os.listdir(images_path)):
    image = Image.open(os.path.join(images_path, file)).copy()
    image_hash = str(average_hash(image))
    if image_hash not in unique_images:
        unique_images.add(image_hash)
        image.save(os.path.join(unique_images_path, file), 'PNG')


import os
from PIL import Image
from imagehash import average_hash
from tqdm import tqdm

dir_image = 'images'
dir_similarity = "similarity"

if not os.path.exists(dir_similarity):
    os.makedirs(dir_similarity)

image = Image.open("image.jpg").copy()
image_hash = average_hash(image)

for file in tqdm(os.listdir(dir_image)):
    if file != "image.jpg":
        img = Image.open(os.path.join(dir_image, file)).copy()
        img_hash = average_hash(img)
        if abs(img_hash - image_hash) < 10:
            img.save(os.path.join(dir_similarity, file), 'PNG')



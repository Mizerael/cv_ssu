import cv2
import numpy as np
import os
from tqdm import tqdm

images_dir = 'images'
find_dir = 'find'

if not os.path.exists(find_dir):
    os.makedirs(find_dir)


def search_image(image_path, template_image_path, output_image_path):
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
    w, h = template_image.shape[::-1]

    res = cv2.matchTemplate(image_gray, template_image,cv2.TM_CCOEFF_NORMED)
    threshold = 0.6
    loc = np.where( res >= threshold)
    if len(loc[0]) > 0:
        for pt in zip(*loc[::-1]):
            cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        cv2.imwrite(output_image_path, image)


template_image_path = 'template.jpg'

for file in tqdm(os.listdir(images_dir)):
    main_image_path = os.path.join(images_dir, file)
    output_image_path = os.path.join(find_dir, file)
    search_image(main_image_path, template_image_path, output_image_path)


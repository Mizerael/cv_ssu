from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

photos_path = Path("Cv")
photo_list = [photo for photo in photos_path.iterdir() if photo.suffix == ".jpg"]

mean_brightness_list = []
ev =  np.linspace(-2.0, 2.0, 40)
for photo in photo_list:
    img = cv2.imread(str(photo))
    x, y = img.shape[0:2]
    batches = [
        math.log2(np.mean(img[i * 64 : (i + 1) * 64, j * 64 : (j + 1) * 64]))
        for i in range(x // 64 + (x % 64 > 0))
        for j in range(y // 64 + (y % 64 > 0))
    ]
    mean_brightness = np.mean(batches)
    mean_brightness_list.append(mean_brightness)

plt.figure(figsize=(10, 6))
plt.plot(
    ev, mean_brightness_list, label="root mean brightness of every photo vs. dynamic range"
)
plt.xlabel('EV')
plt.grid(True)
plt.legend()
plt.savefig("dynamic_range.png")

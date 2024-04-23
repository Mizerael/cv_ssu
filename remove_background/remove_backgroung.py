import os
from PIL import Image, ImageFilter

dir_image = 'image'
dir_treshold = "treshold"
dir_red = "red"
dir_green = "green"
dir_blue = "blue"
dir_updated_images = 'updated_images'
dir_erosion = 'erosion'
dir_dilation = 'dilation'
dir_mask = "mask"

if not os.path.exists(dir_treshold):
    os.makedirs(dir_treshold)
if not os.path.exists(dir_blue):
    os.makedirs(dir_blue)
if not os.path.exists(dir_green):
    os.makedirs(dir_green)
if not os.path.exists(dir_red):
    os.makedirs(dir_red)
if not os.path.exists(dir_erosion):
    os.makedirs(dir_erosion)
if not os.path.exists(dir_dilation):
    os.makedirs(dir_dilation)
if not os.path.exists(dir_mask):
    os.makedirs(dir_mask)
if not os.path.exists(dir_updated_images):
    os.makedirs(dir_updated_images)

def erode(cycles, image):
    for _ in range(cycles):
         image = image.filter(ImageFilter.MinFilter(3))
    return image


def dilate(cycles, image):
    for _ in range(cycles):
         image = image.filter(ImageFilter.MaxFilter(3))
    return image

treshold = 208

for file in os.listdir(dir_image):
    img = Image.open(os.path.join(dir_image, file)).copy()
    img.load()
    img_gray = img.convert('L')
    colors = img.split()
    selected_color = colors[2]
    img_treshold = selected_color.point(lambda x: 255 if x < treshold else 0).convert("1")
    # erode_img = erode(1, img_treshold)
    dilate_img = dilate(5, img_treshold)
    mask = erode(5, dilate_img)
    mask = mask.convert("L").filter(ImageFilter.BoxBlur(3))
    blank = img.point(lambda _: 0)
    segmented_img = Image.composite(img, blank, mask)
    colors[2].save(os.path.join(dir_blue, file), 'PNG')
    colors[1].save(os.path.join(dir_green, file), 'PNG')
    colors[0].save(os.path.join(dir_red, file), 'PNG')
    img_treshold.save(os.path.join(dir_treshold, file), 'PNG')
    # erode_img.save(os.path.join(dir_erosion, file), 'PNG')
    dilate_img.save(os.path.join(dir_dilation, file), 'PNG')
    mask.save(os.path.join(dir_mask, file), 'PNG')
    segmented_img.save(os.path.join(dir_updated_images, file), 'PNG')



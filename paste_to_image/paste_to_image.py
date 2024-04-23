import cv2

ALPHA = 0.7

overlay = cv2.imread('paste_to_image/image.png', cv2.IMREAD_UNCHANGED)
background = cv2.imread('paste_to_image/background.png')
h, w, _ = overlay.shape
background = cv2.resize(background, (w, h))


height, width = overlay.shape[:2]
for y in range(height):
    for x in range(width):
        overlay_color = overlay[y, x, :3]
        overlay_alpha = overlay[y, x, 3] / 255
        background_color = background[y, x]
        composite_color = background_color * (1 - overlay_alpha) + overlay_color * overlay_alpha
        background[y, x] = composite_color

cv2.imwrite('paste_to_image/combined.png', background)



import numpy as np
import cv2

from utils import otsu_threshold, conv, gaussian

import matplotlib.pyplot as plt

img = cv2.imread('open.jpg', cv2.IMREAD_GRAYSCALE)

# Create a blurred mask from otsu threshold
otsu_t = otsu_threshold(img)
alpha = np.where(img > otsu_t, 1.0, 0.0)

ksize = 21
alpha = conv(alpha, gaussian(ksize))
mask = alpha*255

plt.imshow(mask, cmap='gray')
plt.show()

# Save to file mask
cv2.imwrite('mask.jpg', mask.astype(np.uint8))

# Read image and background with colors
img_color = cv2.imread('open.jpg', cv2.IMREAD_COLOR)
bg = cv2.imread('cute-cat-resized.jpg', cv2.IMREAD_COLOR)

# Extend mask for colors
alpha = np.repeat(alpha[:, :, np.newaxis], 3, 2)

# Move the hand up a little (along with the mask)
dy = -180
alpha = np.roll(alpha, dy, 0)
img_color = np.roll(img_color, dy, 0)

# Lerp
combined = alpha * img_color + (1.0-alpha) * bg
combined_rgb = cv2.cvtColor(combined.astype(np.uint8), cv2.COLOR_BGR2RGB)

plt.imshow(combined_rgb, cmap='gray')
plt.show()

# Save to file combined result
cv2.imwrite('combined.jpg', combined)

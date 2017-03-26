import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import *

persp = cv2.imread('persp.jpg', cv2.IMREAD_COLOR)
ortho = cv2.imread('ortho.jpg', cv2.IMREAD_COLOR)

pts_persp = np.array([(181, 289), (186, 217), (459, 176), (451, 340), (173, 391)])
pts_ortho = np.array([(26, 238), (28, 149), (375, 88), (359, 300), (25, 357)])

# Show matching points as polylines
imshow(addPoints(persp, pts_persp))
imshow(addPoints(ortho, pts_ortho))

# Prepare viewport to account for projected pixels
delta_size = (200, 100)
offset = (60, 100)

new_size = (ortho.shape[1] + delta_size[0], ortho.shape[0] + delta_size[1])
pts_ortho_t = pts_ortho + np.array(offset)

# Compute original and projected images in viewport
original = translate(ortho, offset, new_size)
warped = warpImage(persp, pts_persp, pts_ortho_t, new_size)

imshow(original)
imshow(warped)

lerped = np.where(original == 0, warped, original)
imshow(lerped)

cv2.imwrite('computed.jpg', lerped)

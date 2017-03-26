import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import *

bg = cv2.imread('shop_window.jpg', cv2.IMREAD_COLOR)
poster = cv2.imread('poster.jpg', cv2.IMREAD_COLOR)

# Shop window corners in original image
pts_win = np.array([(114, 89), (457, 25), (452, 366), (125, 255)])

# Show window outline
imshow(addPoints(bg, pts_win))

bg_w, bg_h = float(bg.shape[1]), float(bg.shape[0])
p_w, p_h = float(poster.shape[1]), float(poster.shape[0])

bg_ar = bg_w / bg_h
poster_ar = p_w / p_h

# Poster size and offset relative to window viewport
rel_offset = (0.15, 0.15)
rel_height = 0.7
rel_width = rel_height * poster_ar / bg_ar

# Window size relative to poster viewport
vp_size = np.array([p_w/rel_width, p_h/rel_height])
vp_offset = np.array([-rel_offset[0]*vp_size[0], -rel_offset[1]*vp_size[1]])

pts_poster = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * vp_size + vp_offset

poster_rgba = cv2.cvtColor(poster, cv2.COLOR_BGR2BGRA)
warped_poster = warpImage(poster_rgba, pts_poster, pts_win, (bg.shape[1], bg.shape[0]))

lerped = np.where(warped_poster[:, :, 3:] == 0, bg, warped_poster[:, :, :3])
imshow(lerped)

cv2.imwrite('shop_window_poster.jpg', lerped)


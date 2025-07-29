# BGR, BGRA, Ahlpha 채널

import cv2
import numpy as np

# 기본 값 옵션
img = cv2.imread('../img/like_lenna.png')

#IMREAD_COLOR 옵션(BGR)
bgr = cv2.imread('../img/like_lenna.png', cv2.IMREAD_COLOR)

#IMREAD_UNCHANGED 옵션(BGRA)
bgra = cv2.imread('../img/like_lenna.png', cv2.IMREAD_UNCHANGED)

# shape
print("default", img.shape, "color", bgr.shape, "alpha", bgra.shape)

cv2.imshow('img', img)
cv2.imshow('bgr', bgr)
cv2.imshow('alpha', bgra[:,:,3])

cv2.waitKey(0)
cv2.destroyAllWindows()
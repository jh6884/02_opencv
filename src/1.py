import cv2
import numpy as np
import matplotlib.pylab as plt

img1 = cv2.imread('../img/minhabaek_unsplash.jpg')
img2 = cv2.imread('../img/mehrnaz_unsplash.jpg')

px = img1[5, 5]
chromakey = img1[:10, :10, :]
hsv_chroma = cv2.cvtColor(chromakey, cv2.COLOR_BGR2HSV)
# chroma_h = hsv_chroma[:,:,0]
# print(chroma_h)
hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
print(hsv_img)
# print(chromakey)
# print(px)
# BGR 이미지를 회색조로 변환하기

import cv2
import numpy as np

img = cv2.imread('../img/like_lenna.png')

# 첫 번째 방법 (평균값을 이용해 직접 구현하는 방법)
img2 = img.astype(np.uint16) # dtype 변경
b,g,r = cv2.split(img2) # 채널별로 분리
gray1 = ((b + g + r)/3).astype(np.uint8) # 평균값 연산 후 dtype 변경

# 두 번째 방법 (모듈 내 함수 사용)
gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # BGR을 Gray scale로 변경
gray3 = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR) # Gray scale을 BGR로 변경


cv2.imshow('original', img)
cv2.imshow('gray1', gray1)
cv2.imshow('gray2', gray2)
cv2.imshow('gray3', gray3)

cv2.waitKey(0)
cv2.destroyAllWindows
'''
일정한 영역 내에서 극단적으로 어둡거나 밝은 부분이 있으면 노이즈가 생겨 원하는 결과를 얻을 수 없게 됩니다. 이 문제를 피하기 위해서 어떤 영역이든 지정된 제한 값(아래 코드에서 clipLimit 파라미터)을 넘으면 그 픽셀은 다른 영역에 균일하게 배분하여 적용합니다. 이러한 평탄화 방식을 CLAHE라고 합니다.

clahe = cv2.createCLAHE(clipLimit, tileGridSize)
clipLimit: 대비(Contrast) 제한 경계 값, default=40.0
tileGridSize: 영역 크기, default=8 x 8
clahe: 생성된 CLAHE 객체

clahe.apply(src): CLAHE 적용
src: 입력 이미지
'''

import cv2
import numpy as np
import matplotlib.pylab as plt

# 이미지를 읽고 YUV 컬러스페이스로 변경
img = cv2.imread('../img/like_lenna.png')
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# 밝기 채널에 대해서 평탄화 적용
img_eq = img_yuv.copy()
img_eq[:,:,0] = cv2.equalizeHist(img_eq[:,:,0])
img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YUV2BGR)

# 밝기 채널에 대해서 CLAHE 적용
img_clahe = img_yuv.copy()
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) #CLAHE 생성
img_clahe[:,:,0] = clahe.apply(img_clahe[:,:,0]) #CLAHE 적용
img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)

# 결과 출력
cv2.imshow('Before', img)
cv2.imshow('CLAHE', img_clahe)
cv2.imshow('equalizeHist', img_eq)
cv2.waitKey()
cv2.destroyAllWindows()
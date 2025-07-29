'''
정규화(Normalization)
이미지 작업에서도 정규화가 필요한 경우가 있습니다. 특정 영역에 몰려 있는 경우 화질을 개선하기도 하고, 이미지 간의 연산 시 서로 조건이 다른 경우 같은 조건으로 만들기도 합니다. OpenCV는 cv2.normalize()라는 함수로 정규화를 제공합니다.

dst = cv2.normalize(src, dst, alpha, beta, type_flag)
src: 정규화 이전의 데이터
dst: 정규화 이후의 데이터
alpha: 정규화 구간 1
beta: 정규화 구간 2, 구간 정규화가 아닌 경우 사용 안 함
type_flag: 정규화 알고리즘 선택 플래그 상수
'''

import cv2
import numpy as np
import matplotlib.pylab as plt

# 그레이 스케일로 영상 읽기
img = cv2.imread('../img/minhabaek_unsplash.jpg', cv2.IMREAD_GRAYSCALE)

# 직접 연산한 정규화
img_f = img.astype(np.float32)
img_norm = ((img_f - img_f.min()) * (255) / (img_f.max() - img_f.min()))
img_norm = img_norm.astype(np.uint8)

# OpenCV API를 이용한 정규화
img_norm2 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# 히스토그램 계산
hist = cv2.calcHist([img], [0], None, [256], [0, 255])
hist_norm = cv2.calcHist([img_norm], [0], None, [256], [0, 255])
hist_norm2 = cv2.calcHist([img_norm2], [0], None, [256], [0, 255])

cv2.imshow('Before', img)
cv2.imshow('Manual', img_norm)
cv2.imshow('cv2.normalize()', img_norm2)

hists = {'Before' : hist, 'Manual':hist_norm, 'cv2.normalize()':hist_norm2}
for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1,3,i+1)
    plt.title(k)
    plt.plot(v)
plt.show()
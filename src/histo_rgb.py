'''
OpenCV에서는 cv2.calcHist()라는 함수를 통해 히스토그램을 구현할 수 있습니다.

cv2.calHist(img, channel, mask, histSize, ranges)
img: 이미지 영상, [img]처럼 리스트로 감싸서 전달
channel: 분석 처리할 채널, 리스트로 감싸서 전달 - 1 채널: [0], 2 채널: [0, 1], 3 채널: [0, 1, 2]
mask: 마스크에 지정한 픽셀만 히스토그램 계산, None이면 전체 영역
histSize: 계급(Bin)의 개수, 채널 개수에 맞게 리스트로 표현 - 1 채널: [256], 2 채널: [256, 256], 3 채널: [256, 256, 256]
ranges: 각 픽셀이 가질 수 있는 값의 범위, RGB인 경우 [0, 256]
'''

import cv2
import numpy as np
import matplotlib.pylab as plt

# 이미지를 읽고 출력하기
img = cv2.imread('../img/like_lenna.png')
cv2.imshow('img', img)

# 컬러값 채널 분리하기
channels = cv2.split(img)
colors = ('b', 'g', 'r')

# 히스토그램 계산하고 그리기
for (ch, color) in zip (channels, colors):
    hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)

plt.show()
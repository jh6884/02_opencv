# 전역 스레시홀딩

import cv2
import numpy as np
import matplotlib.pylab as plt

img = cv2.imread('../img/like_lenna.png', cv2.IMREAD_GRAYSCALE) # 이미지를 그레이 스케일로 읽기

# numpy api로 바이너리 이미지 만들기
thresh_np = np.zeros_like(img) # 원본과 동일한 크기의 0으로 채워진 이미지
thresh_np[img > 127] = 255 # 127보다 큰 값만 255로 변경

# opencv api로 바이너리 이미지 만들기
ret, thresh_cv = cv2.threshold(img, 63, 255, cv2.THRESH_BINARY)
print(ret) # 127.0 바이너리 이미지에 사용된 문턱 값 반환

# 원본과 결과물을 matplotlib으로 출력
imgs = {'Original': img, 'Numpy api': thresh_np, 'cv2.threshold': thresh_cv}
for i, (key, value) in enumerate(imgs.items()):
    plt.subplot(1, 3, i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])

plt.show()

''' cv2.threshold() 사용법
ret, out = cv2.threshold(img, threshold, value, type_flag)
img: 변환할 이미지
threshold: 스레시홀딩 임계값
value: 임계값 기준에 만족하는 픽셀에 적용할 값
type_flag: 스레시홀딩 적용 방법
'''
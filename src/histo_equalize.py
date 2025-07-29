'''
평탄화(Equalization)
앞서 설명한 정규화는 분포가 한곳에 집중되어 있는 경우에는 효과적이지만 그 집중된 영역에서 멀리 떨어진 값이 있을 경우에는 효과가 없습니다. 이런 경우 평탄화가 필요합니다. 평탄화는 각각의 값이 전체 분포에 차지하는 비중에 따라 분포를 재분배하므로 명암 대비를 개선하는 데 효과적입니다. 

이미지의 히스토그램이 특정 영역에 너무 집중되어 있으면 명암 대비가 낮아 좋은 이미지라고 할 수 없습니다. 전체 영역에 골고루 분포가 되어 있을 때 좋은 이미지라고 할 수 있습니다. 아래 히스토그램을 보면 좌측처럼 특정 영역에 집중되어 있는 분포를 오른쪽처럼 골고루 분포하도록 하는 작업을 히스토그램 평탄화(Histogram Equalization)라고 합니다. (출처: opencv-python.readthedocs.io)

dst = cv2.equalizeHist(src, dst)
src: 대상 이미지, 8비트 1 채널
dst(optional): 결과 이미지
'''

# 회색조 이미지에 평탄화 적용 (histo_equalize.py)

import cv2
import numpy as np
import matplotlib.pylab as plt

# 대상 영상으로 그레이 스케일로 읽기
img = cv2.imread('../img/yate.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape[:2]

# 평탄화 연산을 직접 적용
hist = cv2.calcHist([img], [0], None, [256], [0, 256]) #히스토그램 계산
cdf = hist.cumsum() # 누적 히스토그램 
cdf_m = np.ma.masked_equal(cdf, 0) # 0(zero)인 값을 NaN으로 제거
cdf_m = (cdf_m - cdf_m.min()) /(rows * cols) * 255 # 평탄화 히스토그램 계산
cdf = np.ma.filled(cdf_m,0).astype('uint8') # NaN을 다시 0으로 환원
print(cdf.shape)
img2 = cdf[img] # 히스토그램을 픽셀로 맵핑

# OpenCV API로 평탄화 히스토그램 적용
img3 = cv2.equalizeHist(img)

# 평탄화 결과 히스토그램 계산
hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([img3], [0], None, [256], [0, 256])

# 결과 출력
cv2.imshow('Before', img)
cv2.imshow('Manual', img2)
cv2.imshow('cv2.equalizeHist()', img3)
hists = {'Before':hist, 'Manual':hist2, 'cv2.equalizeHist()':hist3}
for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1,3,i+1)
    plt.title(k)
    plt.plot(v)
plt.show()
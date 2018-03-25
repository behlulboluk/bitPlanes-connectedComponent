import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../imgs/CCimg.png', 0)
ret, threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

element = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
temp = cv2.dilate(threshold, element)

gauss = cv2.adaptiveThreshold(temp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)

ret1, labels = cv2.connectedComponents(gauss, connectivity=8)

label_cnt = np.uint8(255*labels/np.max(labels))
shiny_arr = 255*np.ones_like(label_cnt)

label_img = cv2.merge([label_cnt, shiny_arr, shiny_arr])
label_img = cv2.cvtColor(label_img, cv2.COLOR_HSV2BGR)

for i in range(1, np.max(labels)):
    plt.imshow(label_img)
    plt.title(i)
    plt.xticks([]), plt.yticks([])
plt.show()

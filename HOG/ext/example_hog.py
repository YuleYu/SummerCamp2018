import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import math

cellSize = 40
cellWidth = cellSize/2

img = cv.imread('/Users/yule/Desktop/fig.png')
hog = cv.HOGDescriptor((48,48), (16, 16), (8, 8), (8, 8), 9)
h = hog.compute(img);

image = np.zeros([cellSize * 10, cellSize * 10])

for i in range(10):
    for j in range(10):
        angle = 0
        for mag in range(9):
            angle_radian = math.radians(angle)
            x1 = int(i * cellSize + h[100*i+10*j+mag] * cellWidth * math.cos(angle_radian) + cellWidth)
            y1 = int(j * cellSize + h[100*i+10*j+mag] * cellWidth * math.sin(angle_radian) + cellWidth)
            x2 = int(i * cellSize - h[100*i+10*j+mag] * cellWidth * math.cos(angle_radian) + cellWidth)
            y2 = int(j * cellSize - h[100*i+10*j+mag] * cellWidth * math.sin(angle_radian) + cellWidth)
            cv.line(image, (y1, x1), (y2, x2),int(255*math.sqrt(mag)))
            angle += 20

plt.imshow(image, cmap=plt.cm.gray)
plt.show()
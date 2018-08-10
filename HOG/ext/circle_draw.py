# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 20:21:07 2018

@author: jiangjiechu
"""


import cv2
import numpy as np

drawing = False #如果按下鼠标，则为true
mode = True #如果是 True 则画矩形。按 m 键变成绘制曲线。

#鼠标回调函数
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    #当按下左键时返回起始位置坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
     #当鼠标左键按下并移动时是绘制图形   
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
            else:
                cv2.circle(img,(x,y),5,(0,0,255),-1)
     #鼠标松开停止绘画           
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
        else:
            cv2.circle(img,(x,y),5,(0,0,255),1)

img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
size()
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break

cv2.destroyAllWindows()
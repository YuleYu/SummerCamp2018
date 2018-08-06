# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 12:40:25 2018

@author: jiangjiechu
"""


import cv2
import numpy as np
from copy  import deepcopy

drawing = False #如果按下鼠标，则为true
mode = True #如果是 True 则画矩形。按 m 键变成绘制曲线。

def DrawRectangle(event,x,y,flags,param):
    #Function that draw rectangle for region cropping
    global ix,iy,drawing,frame_tmp,frame,x_final,y_final
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            frame_tmp = deepcopy(frame)
            x1 = x-((x-ix)%8)
            y1 = y-((y-iy)%8)
            cv2.rectangle(frame_tmp,(ix,iy),(x1,y1),(0,255,0),0)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        frame_tmp = deepcopy(frame)
        x1 = x-((x-ix)%8)
        y1 = y-((y-iy)%8)
        cv2.rectangle(frame_tmp,(ix,iy),(x1,y1),(255,0,0),0)
        x_final,y_final = x1,y1
        
def GetRegion(img: object) -> object:
    #Function performing region cropping
    global frame_tmp,frame
    frame = deepcopy(img)
    frame_tmp = deepcopy(frame)
    cv2.setMouseCallback('img',DrawRectangle)
    
    while(True):
        cv2.imshow('img',frame_tmp)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    
# home = "/Users/yule/Desktop/basketball"
# cap = cv2.VideoCapture(home+"/video/dongdan_4_1_04011530.mp4")
# success,frame = cap.read()
# cv2.namedWindow('img')
#
# GetRegion(frame)
# form = "%5d\t%5d\n"
# f=open(home+"target_index.txt",'w')
# f.write(form %( ix, x_final))
# f.write(form % (iy, y_final))
# f.close()
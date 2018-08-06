# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 15:34:33 2018

@author: jiangjiechu
"""

import numpy as np
import cv2

cap = cv2.VideoCapture(home+"/video/dongdan_4_1_04011530.mp4")
while (cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
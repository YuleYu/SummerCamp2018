# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 19:43:27 2018

@author: jiangjiechu
"""
import cv2
from numpy import *

def nothing(x):
    pass

fppc = open(home+'video2.ppc','r')
fpng = "".join(fppc.readline().split())
packed_pic = cv2.imread(fpng,0)
t_start,t_end = list(map(int,fppc.readline().split()))
fps = int(fppc.readline())
size_x,size_y = list(map(int,fppc.readline().split()))
n_col = int(fppc.readline())
n_row = (t_end - t_start)*fps/n_col
stride = 20
n_stride = int(n_row/stride)
cv2.namedWindow('image')
cv2.createTrackbar('t1','image',0,n_stride-1,nothing)
cv2.createTrackbar('t2','image',0,stride-1,nothing)

second_count = 0
row_id = second_count
col_id = second_count % n_col

tmp_pic = packed_pic[row_id*size_y:(row_id+5)*size_y,:]#col_id*size_x:(col_id+25)*size_x]
while(1):
    t1=cv2.getTrackbarPos('t1','image')
    t2=cv2.getTrackbarPos('t2','image')
    frame_count = t1*stride+t2
    row_id = frame_count
    col_id = frame_count % n_col
    tmp_pic = packed_pic[row_id*size_y:min((row_id+stride)*size_y,(n_row-1)*size_y),:]#col_id*size_x:(col_id+25)*size_x]
    cv2.imshow('image',tmp_pic)
    k=cv2.waitKey(1)&0xFF
    if k==27:
        break
    

#销毁窗口        
cv2.destroyAllWindows()
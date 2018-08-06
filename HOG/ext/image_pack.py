# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:08:29 2018

@author: jiangjiechu
"""
import cv2
from copy import deepcopy
from numpy import *
"""压缩图片格式:
    压缩图片为拼接图,由描述文件和小图块阵列文件构成
    阵列拼接图片:固定的列数n_column,图片依次向下补充
    描述文件(后缀.ppc):
        第一行:阵列图片文件名
        第二行:起止时间段
        第三行:帧率
        第四行:分辨率
        第五行:列数    
"""
def ShowPic(pic):
    while(1):   
        cv2.imshow('image',pic)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

def GenPackedPic(home,t_start,t_end,fps,n_col):
    f_target_index = open(home+"target_index.txt",'r')
    ix, x_final,iy,y_final = list(map(int, f_target_index.read().split()))
    cap = cv2.VideoCapture(home+"/video/dongdan_4_1_04011530.mp4")
    size_x = x_final-ix
    size_y = y_final-iy
    f_target_index.close()
    n_frames = (t_end-t_start)*fps
    n_row = int(n_frames /n_col  )
    out_pic = zeros((size_y*n_row,size_x*n_col),uint8)
    frame_count = 0
    while (cap.isOpened()):
        success,frame = cap.read()
        frame_count = frame_count +1
        if success == True :
            if frame_count < t_start*fps:
                continue
            if frame_count >= t_start*fps :  
                if frame_count <=t_end * fps :
                    if frame_count % (60*25) == 0:
                        print("%d minute\n"%(frame_count/(60*25)))
                    row_id = int((frame_count-t_start*fps-1) / n_col)
                    col_id = int((frame_count-t_start*fps-1) % n_col)
                    target_region = cv2.cvtColor(frame[iy:y_final,ix:x_final,:],cv2.COLOR_BGR2GRAY)
                    out_pic[row_id*size_y:(row_id+1)*size_y,col_id*size_x:(col_id+1)*size_x]= target_region
                else:
                    break
        else:
            break
    cap.release()
    return out_pic
    
def WritePackedPic(home,filename,out_pic,t_start,t_end,n_col):
    f_target_index = open(home+"target_index.txt",'r')
    ix, x_final,iy,y_final = list(map(int, f_target_index.read().split()))
    size_x = x_final-ix
    size_y = y_final-iy
    f_target_index.close()
    cv2.imwrite(home+filename+'.png',out_pic)
    f_description = open(home+filename+'.ppc','w')
    f_description.write(home+filename+'.png\n')
    f_description.write('%5d\t%5d\n'%(t_start,t_end))
    f_description.write('%5d\n'%fps)
    f_description.write('%5d\t%5d\n'%(size_x,size_y))
    f_description.write('%5d\n'%n_col)
    f_description.close()


out_pic=GenPackedPic(home,t_start,t_end,fps,n_col)
WritePackedPic(home,'video3',out_pic,t_start,t_end,n_col)

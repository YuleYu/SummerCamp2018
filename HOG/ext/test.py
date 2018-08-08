from HOG.ext import functions as fun
from HOG.ext import functions_jiang as fun_j
import cv2
import numpy as np
import datetime
import random
import matplotlib.pyplot as plt
import os
from LR.ext.Functions import *

home = 'C:/Users/peter/Documents/GitHub/SummerCamp2018/HOG/'
t_start,t_end,fps,n_col,size_y,size_x,packed_img = fun.ReadPackedImg('result/video1',home)
#test_pic = fun.ShowFrame(packed_img,size_y,size_x,329,5)
test_pic = fun.GetOneFrame(packed_img,size_y,size_x,329,n_col)
# hog = cv2.HOGDescriptor((size_y,size_x),(16,16),(8,8),(8,8),9)
# h = hog.compute(test_pic)
# n_cells = int(np.sqrt(h.size/9))
# h = h.reshape((n_cells,n_cells,9))
h = fun.HOGCalc(test_pic,8,9)
h_img = fun.HOG_pic(test_pic,h)
# h_img = fun.HOG_pic_cv2(test_pic,h)
#采集正样本
f_positive = open(home+'result/video1_pos.txt','r')
s_pos = []
for entry in f_positive:
    f_beg,f_end = list(map(int,entry.split()))
    for i in range(f_beg,f_end+1):
        s_pos.append((i,fun.GetOneFrame(packed_img,size_y,size_x,i,n_col)))
f_positive.close()

hog_pos = CalcSample(s_pos,1)

#提取负样本,暂定从f_beg向后取5~50内的帧,可形成405~4050个大小的负样本集合
#用neg_per_shot控制一次假进球帧后面采集多少负样本,neg_per_shot越大,静止帧数越多
f_neg = open(home+'result/video1_neg.txt','r')
s_neg = []
for entry in f_neg:
    f_beg,f_end = list(map(int,entry.split()))
    # print("%d~%d is processing:\n"%(f_beg,f_end))
    for i in range(f_beg,f_beg+5):
        s_neg.append((i,fun.GetOneFrame(packed_img,size_y,size_x,i,n_col)))
f_neg.close()
total_neg = len(s_neg)
print(total_neg)


hog_neg = CalcSample(s_neg,0)

#计算正样本中心center_pos
center_pos = sum(hog_pos)/hog_pos.shape[0]

#计算正负样本到center的距离
dist_pos = np.zeros((hog_pos.shape[0]))
for i in range(hog_pos.shape[0]):
    dist_pos[i] = np.sqrt(sum((hog_pos[i]-center_pos).reshape(hog_pos[i].size)**2))
dist_neg = np.zeros((hog_neg.shape[0]))
for i in range(hog_neg.shape[0]):
    dist_neg[i] = np.sqrt(sum((hog_neg[i]-center_pos).reshape(hog_neg[i].size)**2))

train,test = BuildSet(hog_pos,hog_neg,0.2)

def LRTest(w,test,fun=np.log10):
    roc_dots = []
    for sample in test:
        y = sample[1]
        x = sample[2:sample.size]
        x = np.append(x,1)
        value = np.dot(w,x)
        roc_dots.append([sample[0],y,value])
    roc_dots = np.array(roc_dots)
    roc_dots = roc_dots[np.lexsort(roc_dots.T)]
    (x,y) = (0,0)
    roc_curve = [[x,y]]
    n_pos = sum(test[:,1])
    n_neg = test.shape[0] - n_pos
    for elem in roc_dots:
        if elem[1] == 1:
            x += 1.0 / n_pos
        elif elem[1] == 0:
            y += 1.0 / n_neg
        roc_curve.append([x,y])
    roc_curve = np.array(roc_curve)
    # plt.plot(fun(roc_curve[:,0]),roc_curve[:,1])
    plt.xlim(1e-4,1)
    plt.semilogx(fun(roc_curve[:,0]),roc_curve[:,1])
    return roc_curve

w,curve = LRLearning(train)
LRTest(w,test,fun=(lambda x: x))
LRTest(w,test)


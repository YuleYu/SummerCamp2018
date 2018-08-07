import cv2
import numpy as np
import time
from HOG.ext import functions as fun
import  matplotlib.pyplot as plt

def ReadSample(home,fname,packed_img,size_y,size_x,n_col,lag=5):
    f_positive = open(home+fname+'_pos.txt','r')
    s_pos = []
    for entry in f_positive:
        f_beg,f_end = list(map(int,entry.split()))
        for i in range(f_beg,f_end+1):
            s_pos.append(fun.GetOneFrame(packed_img,size_y,size_x,i,n_col))
    f_positive.close()

    f_neg = open(home+fname+'_neg.txt','r')
    s_neg = []
    for entry in f_neg:
        f_beg,f_end = list(map(int,entry.split()))
        for i in range(f_beg,f_beg+lag):
            print("%d\t"%i)
            s_neg.append(fun.GetOneFrame(packed_img,size_y,size_x,i,n_col))
        print("\n")
    f_neg.close()
    total_neg = len(s_neg)
    print(total_neg)

#计算并绘制ROC曲线
# def GenROC(fpos,fneg):
#     fpos.sort()
#     fneg.sort()
#     i,j=(0,0)
#     x,y=(0.,0.)
#     xypoints = [[x,y]]
#     while i != fpos.size-1 and j != fneg.size-1:
#         if fpos[i] <= fneg[j]:
#             y += 1.0 / fpos.size
#             i += 1
#         elif fpos[i] > fneg[j]:
#             x += 1.0 / fneg.size
#             j += 1
#         xypoints.append([x,y])
#
#     xypoints = np.array(xypoints)
#     plt.plot(np.log10(xypoints[1:-1,0]), xypoints[1:-1,1],'r',xypoints[1:-1,0],xypoints[1:-1,1],'g')
#     plt.show()
#


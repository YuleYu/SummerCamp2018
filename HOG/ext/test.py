from HOG.ext import functions as fun
import cv2
import numpy as np
import datetime
#from HOG.ext.ReadXML import ReadXML
import matplotlib.pyplot as plt

home = 'C:/Users/peter/Documents/GitHub/SummerCamp2018/HOG/'
scriptpath = 'script/'


t_start,t_end,fps,n_col,size_y,size_x,packed_img = fun.ReadPackedImg('result/video1',home)
#test_pic = fun.ShowFrame(packed_img,size_y,size_x,329,5)
test_pic = fun.GetOneFrame(packed_img,size_y,size_x,329,n_col)
# hog = cv2.HOGDescriptor((size_y,size_x),(16,16),(8,8),(8,8),9)
# h = hog.compute(test_pic)
# n_cells = int(np.sqrt(h.size/9))
# h = h.reshape((n_cells,n_cells,9))
# h_img = fun.HOG_pic_cv2(test_pic,h)

f_positive = open(home+'result/video1_pos.txt','r')
s_pos = []
for entry in f_positive:
    f_beg,f_end = list(map(int,entry.split()))
    # print("%d~%d is processing:\n"%(f_beg,f_end))
    for i in range(f_beg,f_end+1):
        print("%d\t"%i)
        s_pos.append(fun.GetOneFrame(packed_img,size_y,size_x,i,n_col))
    print("\n")

f_positive.close()

# hog_pos = []
# start_time = datetime.datetime.now()
# for pic in s_pos:
    # hog_pos.append(fun.HOGCalc(pic,8,9))

# end_time = datetime.datetime.now()
# print((end_time-start_time).seconds)
# hog_pos = np.array(hog_pos)


f_neg = open(home+'result/video1_neg.txt','r')
s_neg = []
for entry in f_neg:
    f_beg,f_end = list(map(int,entry.split()))
    # print("%d~%d is processing:\n"%(f_beg,f_end))
    for i in range(f_beg,f_beg+5):
        print("%d\t"%i)
        s_neg.append(fun.GetOneFrame(packed_img,size_y,size_x,i,n_col))
    print("\n")

f_neg.close()
total_neg = len(s_neg)
print(total_neg)

hog_neg = []
start_time = datetime.datetime.now()
for pic in s_neg:
    hog_neg.append(fun.HOGCalc(pic,8,9))

end_time = datetime.datetime.now()
print((end_time-start_time).seconds)
hog_pos = np.array(hog_neg)

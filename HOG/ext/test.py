from HOG.ext import functions as fun
import cv2
import numpy as np
#from HOG.ext.ReadXML import ReadXML
import matplotlib.pyplot as plt

home = 'C:/Users/peter/Documents/GitHub/SummerCamp2018/HOG/'
scriptpath = 'script/'


t_start,t_end,fps,n_col,size_y,size_x,packed_img = fun.ReadPackedImg('result/video1',home)
#test_pic = fun.ShowFrame(packed_img,size_y,size_x,329,5)
test_pic = fun.GetOneFrame(packed_img,size_y,size_x,329,n_col)
hog = cv2.HOGDescriptor((size_y,size_x),(16,16),(8,8),(8,8),9)
h = hog.compute(test_pic)
n_cells = int(np.sqrt(h.size/9))
h = h.reshape((n_cells,n_cells,9))
h_img = fun.HOG_pic_cv2(test_pic,h)
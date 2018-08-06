import cv2
from copy import deepcopy
from numpy import *

<<<<<<< HEAD
home = "C://Users/peter/Documents/GitHub/SummerCamp2018/HOG/"
#home = "/User/yule/Desktop/basketball"
=======
home = '.'
cap = cv2.VideoCapture(home+"/video/dongdan_4_1_04011530.mp4")
success,frame = cap.read()
>>>>>>> c7f01673a18b939c2367d1de2673d25ff7ad8c20
frame_count=0
frame_interval=2
fps = 25
t_start = 0
t_end = 30*60
n_col = 5

#WriteRegion


print("capture success!")
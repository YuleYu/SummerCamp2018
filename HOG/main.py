import cv2
from copy import deepcopy
from numpy import *

home = '.'
cap = cv2.VideoCapture(home+"/video/dongdan_4_1_04011530.mp4")
success,frame = cap.read()
frame_count=0
frame_interval=2
fps = 25
t_start = 0
t_end = 30*60
n_col = 5

#WriteRegion


print("capture success!")
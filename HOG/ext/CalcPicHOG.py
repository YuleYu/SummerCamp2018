from HOG.ext import functions as fun
import cv2
import numpy as np
import datetime
import random
from HOG.ext.ReadXML import ReadMainXML, ReadXML
import matplotlib.pyplot as plt
import os

script_path = '../script/'
mainScript = script_path + 'main_HOG.xml'
startHOGpic, endHOGpic, showMidResult, video_choice, get_region = ReadMainXML(mainScript)

pic_path = '../pic/'
video_path = '../video/'
result_path = '../result'
pic_name = '{0}.png'.format(video_choice)
script_name = '{0}.xml'.format(video_choice)
video_name = '{0}.mp4'.format(video_choice)

# read in script
script = "{0}{1}".format(script_path, script_name)
pic = "{0}{1}".format(pic_path, pic_name)
video = "{0}{1}".format(video_path, video_name)
usls_img, size_x, size_y, t_start, t_end, n_col, fps, bin = ReadXML(script)

# home = '../result/'
# t_start,t_end,fps,n_col,size_y,size_x,packed_img = fun.ReadPackedImg('video1',home)
packed_img = cv2.imread(pic,0)
test_pic = fun.GetOneFrame(packed_img,size_y,size_x,329,n_col)

# 采集所有样本
samples = []
for i in range((t_end-t_start)*fps-5):
# for i in range(2):
    samples.append((i, fun.GetOneFrame(packed_img, size_y, size_x, i, n_col)))

hog_list = []
for (i,sample) in samples:
    hog = np.array(fun.HOGCalc(sample, 8, bin))
    sample = np.zeros((hog.size+1))
    sample[0] = i
    sample[1:hog.size+2] = hog.reshape(hog.size)
    hog_list.append(sample)
    print(str(i) + ' frame done.')

# 保存 HOG 文件
save_file = result_path + '/' + video_choice + '.dat'
f = open(save_file, 'wb')
for every in hog_list:
    f.writelines(str(every))
f.close()

from HOG.ext.functions import *
import cv2
import numpy as np
import pickle
#使用cv2.hog计算HOG值,比自己手写的快太多;
def CalcHOG(fvideo):
    t_start,t_end,fps,n_col,size_y,size_x,packed_img = ReadPackedImg(fvideo)
    hog = cv2.HOGDescriptor((size_y,size_x),(16,16),(8,8),(8,8),9)
    fhog = open(fvideo + '.hog','wb')
    nframe = int(t_end-t_start)*fps
    hog_size = (size_y/8 - 1)*(size_x/8 - 1)*2*2*9
    h = np.zeros((nframe,hog_size),np.float32)
    for i in range(nframe):
        if i % 6000*fps == 0:
            portion = int(i/nframe*100)
            print("%d%% is completed"%portion)
        test_pic = GetOneFrame(packed_img,size_y,size_x,i,n_col)
        h[i] = hog.compute(test_pic).reshape(hog_size)
    pickle.dump(h,fhog,1)
    fhog.close()

def LoadHOG(fvideo):
    fhog = open(fvideo + '.hog', 'rb')
    h = pickle.load(fhog)
    return h

# unfinished
def CalHOG_yh(imgPath, size_x, size_y, startTime, endTime, colSize, fps, bin):
    block_size = 16
    block_stride = 8
    cell_size = 8
    hogDes = cv2.HOGDescriptor((size_y, size_x), (block_size, block_size), (block_stride,block_stride), (cell_size,cell_size), bin)
    fhog = open(imgPath, 'wb')
    nframe = int(endTime - startTime) * fps
    hog_size = (size_y / cell_size - 1) * (size_x / 8 - 1) * (block_size / cell_size)**2 * bin
    for i in range(nframe):
        pic = GetOneFrame()
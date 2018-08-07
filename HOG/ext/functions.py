# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 14:34:26 2018

@author: jiangjiechu
"""

import cv2
from numpy import *
import time

def GaussianFilter(n):
    filter_list = zeros(n)
    mu = (n+1)/2
    sigma = mu/3
    for i in arange(n):
        filter_list[i]=exp(-square(i+1-mu)/(2*square(sigma)))
    return filter_list/sum(filter_list)
    
def Convolution(data,n):
    filter_list = GaussianFilter(n)
    di = int((n-1)/2)
    data_new = zeros(data.size)
    for i in arange(di,data.size-di):
        #-print("%d\n"%i)
        data_new[i] = sum(data[i-di:i+di+1]*filter_list)
    data_new[0:di]=data[0:di]
    data_new[data.size-di:data.size]=data[data.size-di:data.size]
    return data_new

def CalcCriterion(t_start,t_end,fps,n_col,size_y,size_x,packed_pic):    
    frame_limit = (t_end-t_start)*fps
    criterion = array(zeros((frame_limit),float))
    for frame_count in arange(0,frame_limit):
        row_id = int(frame_count/n_col)
        col_id = frame_count % n_col
        tmp_pic = packed_pic[row_id*size_y:(row_id+1)*size_y,col_id*size_x:(col_id+1)*size_x]
        criterion[frame_count] = sum(tmp_pic)
    return criterion

def FindCandidates(data,threshold,window):
    candidates = zeros(data.size,uint)
    previous = 0
    for i in arange(data.size-window):
        if candidates[i]==0:
            if data[i]<threshold:
                candidates[i:min(data.size-1,i+window)] += 1
                previous = data[i]
        if candidates[i]==1:
            if data[i]<previous:
                candidates[i:min(data.size-1,i+window)] += 1
                candidates[i] += 1
    return candidates

def CountCandidates(data,threshold,window):
    count = 0
    cand = FindCandidates(data,threshold,window)
    for i in arange(cand.size-1):
        if cand[i] == 0 and cand[i+1] ==1:
            count = count + 1
    return count

def ToTime(t):
    s= time.strftime("%M:%S",time.localtime(t))
    tms = t%1
    return "%s:%2d"%(s,tms*100)

def LocateCandidatesByT(data,threshold,window,lag):
    count = 0
    cand = FindCandidates(data,threshold,window)
    frame_list = []
    start = 0
    end = data.size
    for i in arange(cand.size-1):
        if cand[i+1]>cand[i]:
            count = count + 1
            start = i
            end = start + lag
        elif cand[i] ==1 and cand[i+1] == 0:
            frame_list.append([start,end,end-start])
    interval_list = []
    for elem in array(frame_list)/25:
        interval_entry = []
        for t in elem:
            interval_entry.append(ToTime(t))
        interval_list.append(interval_entry)
    return interval_list

def LocateCandidatesByF(data,threshold,window,lag):
    count = 0
    cand = FindCandidates(data,threshold,window)
    frame_list = []
    start = 0
    end = data.size
    for i in arange(cand.size-1):
        if cand[i+1]>cand[i]:
            count = count + 1
            start = i
            end = start + lag
        elif cand[i] ==1 and cand[i+1] == 0:
            frame_list.append([start,end,end-start])
    return frame_list

# def draw_circle(event,x,y,flags,param):
#     global ix,iy,drawing,mode
#     #当按下左键时返回起始位置坐标
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix,iy = x,y
#      #当鼠标左键按下并移动时是绘制图形
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing == True:
#             if mode == True:
#                 cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
#             else:
#                 cv2.circle(img,(x,y),5,(0,0,255),-1)
#      #鼠标松开停止绘画
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         if mode == True:
#             cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
#         else:
#             cv2.circle(img,(x,y),5,(0,0,255),1)

def Gradient(img):
    img = img.astype(float)
    size_y,size_x = img.shape
    dx = zeros((size_y,size_x))
    dy = zeros((size_y,size_x))
    dx[:,1:size_x-1] = (img[:,2:size_x]-img[:,0:size_x-2])
    dx[:,0]=dx[:,1]
    dx[:,size_x-1] = dx[:,size_x-2]
    dy[1:size_y-1,:] = (img[2:size_y,:]-img[0:size_y-2,:])
    dy[0,:]=dy[1,:]
    dy[size_y-1,:]=dy[size_y-2,:]
    return [dx,dy]

def HOGCalc(img,cell_width, n):
    interval = 180/n
    half_cell_width = cell_width / 2
    eps = 0.001
    (dx,dy) = Gradient(img)
    mag = sqrt(dx*dx+dy*dy)
    ang = cv2.phase(dx, dy, angleInDegrees=True) % 180
    size_y,size_x = img.shape
    # bins = zeros((n,size_y,size_x))
    bin_id = (floor(ang/interval)%n).astype(int16)
    hog = zeros((int(floor(size_y/cell_width))+2,int(floor(size_x/cell_width))+2,n))
    for i in arange(size_y):

        for j in arange(size_x):
            cell_i = int(floor((i-half_cell_width)/cell_width)) + 1
            cell_j = int(floor((j-half_cell_width)/cell_width)) + 1
            cell_id = bin_id[i][j]
            cell_id2= (cell_id+1)%n
            #Calculate HOG in adjacent cells and bins
            w_i =(i/cell_width-cell_i)% 1
            w_j =(j/cell_width-cell_j)% 1


            w_id = ang[i][j]/interval-cell_id
            # print("%f,%f,%f\n"%(w_i,w_j,w_id))
            hog[cell_i,cell_j,cell_id] = hog[cell_i,cell_j,cell_id] + mag[i][j]*(1-w_i)*(1-w_j)*(1-w_id)
            hog[cell_i,cell_j,cell_id2] = hog[cell_i,cell_j,cell_id2] + mag[i][j]*(1-w_i)*(1-w_j)*(w_id)
            hog[cell_i+1,cell_j,cell_id] = hog[cell_i+1,cell_j,cell_id] + mag[i][j]*(w_i)*(1-w_j)*(1-w_id)
            hog[cell_i+1,cell_j,cell_id2] = hog[cell_i+1,cell_j,cell_id2] + mag[i][j]*(w_i)*(1-w_j)*(w_id)
            hog[cell_i,cell_j+1,cell_id] = hog[cell_i,cell_j+1,cell_id] + mag[i][j]*(1-w_i)*(w_j)*(1-w_id)
            hog[cell_i,cell_j+1,cell_id2] = hog[cell_i,cell_j+1,cell_id2] + mag[i][j]*(1-w_i)*(w_j)*(w_id)
            hog[cell_i+1,cell_j+1,cell_id] = hog[cell_i+1,cell_j+1,cell_id] + mag[i][j]*(w_i)*(w_j)*(1-w_id)
            hog[cell_i+1,cell_j+1,cell_id2] = hog[cell_i+1,cell_j+1,cell_id2] + mag[i][j]*(w_i)*(w_j)*(w_id)
    hog1=zeros(hog.shape,float)
    for i in arange(1,hog.shape[0]-2):
        for j in arange(1,hog.shape[1]-2):
            blk_norm = sqrt(sum(hog[i:i+2,j:j+2].reshape(4*n)**2)+eps**2)
            hog1[i:i+2,j:j+2,:] = hog[i:i+2,j:j+2,:]/blk_norm
    return hog1[1:-1,1:-1]

def HOG_pic(img,hog):
    size_y,size_x = img.shape
    cell_size = 32
    cell_width = cell_size/2
    max_mag = hog.max()
    # hog_image = zeros([size_y*4,size_x*4])
    img_shape = (hog.shape[0]*cell_size,hog.shape[1]*cell_size)
    hog_image = cv2.resize(img,img_shape,interpolation=cv2.INTER_NEAREST)
    for x in range(hog.shape[0]):
        for y in range(hog.shape[1]):
            cell_grad = hog[x][y]
            cell_grad /= max_mag
            angle = 0
            angle_gap = 180/(hog.shape[2])
            for magnitude in cell_grad:
                angle_radian =radians(angle)
                x1 = int(x * cell_size- magnitude * cell_width * cos(angle_radian)+cell_width)
                y1 = int(y * cell_size+ magnitude * cell_width * sin(angle_radian)+cell_width)
                x2 = int(x * cell_size+ magnitude * cell_width * cos(angle_radian)+cell_width)
                y2 = int(y * cell_size- magnitude * cell_width * sin(angle_radian)+cell_width)
                cv2.line(hog_image,(y1,x1),(y2,x2),int(255-255*sqrt(magnitude)))
                angle += angle_gap
    return hog_image

def HOG_pic_cv2(img,hog):
    # size_y,size_x = img.shape
    cell_size = 32
    cell_width = cell_size/2
    #max_mag = hog.max()
    max_mag =1
    hog_image = zeros([hog.shape[0]*cell_size,hog.shape[1]*cell_size])
    for x in range(hog.shape[0]):
        for y in range(hog.shape[1]):
            cell_grad = hog[x][y]
            cell_grad /= max_mag
            angle = 0
            angle_gap = 180/(hog.shape[2])
            for magnitude in cell_grad:
                angle_radian = math.radians(angle)
                x1 = int(x * cell_size+ magnitude * cell_width * math.sin(angle_radian)+cell_width)
                y1 = int(y * cell_size+ magnitude * cell_width * math.cos(angle_radian)+cell_width)
                x2 = int(x * cell_size- magnitude * cell_width * math.sin(angle_radian)+cell_width)
                y2 = int(y * cell_size- magnitude * cell_width * math.cos(angle_radian)+cell_width)
                cv2.line(hog_image,(y1,x1),(y2,x2),int(255*math.sqrt(magnitude)))
                angle += angle_gap
    return hog_image
def ReadPackedImg(fname,home):
    fppc = open(home+fname+'.ppc','r')
    fpng = (home+fname+'.png')
    packed_img = cv2.imread(fpng,0)
    t_start,t_end = list(map(int,fppc.readline().split()))
    fps = int(fppc.readline())
    size_x,size_y = list(map(int,fppc.readline().split()))
    n_col = int(fppc.readline())
    return [t_start,t_end,fps,n_col,size_y,size_x,packed_img]
    
def ShowFrame(packed_img,size_y,size_x,n,lag):
    tmp_img = zeros((size_y*lag,size_x*lag),uint8)
    n_col = 5
    row_id = int((n-1)/n_col)
    col_id = int((n-1)%n_col)
    print("%d,%d"%(size_y,size_x))
    tmp_img[0:size_y*lag,0:size_x*lag]=packed_img[row_id*size_y:(row_id+lag)*size_y,:]
    #tmp_img[0:size_y,0:size_x*(n_col-col_id)]=packed_img[row_id*size_y:(row_id+1)*size_y,col_id*size_x:(n_col)*size_x]
    #tmp_img[0:size_y,size_x*(n_col-col_id):lag*size_x] = packed_img[(row_id+1)*size_y:(row_id+2)*size_y,0:(lag+col_id-n_col)*size_x]
    #plt.imshow(tmp_img,cmap=plt.cm.gray)   
    return tmp_img

def CheckBalls(fname):
    t_start,t_end,fps,n_col,size_y,size_x,packed_img = ReadPackedImg(fname)
    crit = CalcCriterion(t_start,t_end,fps,n_col,size_y,size_x,packed_img)
    data = Convolution(crit,3)-Convolution(crit,13)
    locations = LocateCandidatesByF(data,-2000,50,5)
    result = []
    for entry in locations:
        tmp_img=ShowFrame(packed_img,size_y,size_x,entry[0],entry[2])
        #print(entry)
        #plt.imshow(tmp_img,cmap=plt.cm.gray)
        #isScore = input()
        #result.append([entry[0],isScore])
    return locations


def GetOneFrame(pkg_img, Y, X, n, n_col):
    row_id = int(n/n_col)
    col_id = int(n%n_col)
    return pkg_img[row_id*Y:(row_id+1)*Y,col_id*X:(col_id+1)*X]
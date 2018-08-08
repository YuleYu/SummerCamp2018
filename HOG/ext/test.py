from HOG.ext import functions as fun
from HOG.ext import functions_jiang as fun_j
import cv2
import numpy as np
import datetime
import random
from HOG.ext.ReadXML import ReadXML
import matplotlib.pyplot as plt

home = './'
t_start,t_end,fps,n_col,size_y,size_x,packed_img = fun.ReadPackedImg('result/video1',home)
#test_pic = fun.ShowFrame(packed_img,size_y,size_x,329,5)
test_pic = fun.GetOneFrame(packed_img,size_y,size_x,329,n_col)
# hog = cv2.HOGDescriptor((size_y,size_x),(16,16),(8,8),(8,8),9)
# h = hog.compute(test_pic)
# n_cells = int(np.sqrt(h.size/9))
# h = h.reshape((n_cells,n_cells,9))
# h_img = fun.HOG_pic_cv2(test_pic,h)
#采集正样本
f_positive = open(home+'result/video1_pos.txt','r')
s_pos = []
for entry in f_positive:
    f_beg,f_end = list(map(int,entry.split()))
    for i in range(f_beg,f_end+1):
        s_pos.append((i,fun.GetOneFrame(packed_img,size_y,size_x,i,n_col)))
f_positive.close()

def CalcSample(s,y):
    #计算样本的hog,正样本y=1,负样本y=0
    hog_list = []
    start_time = datetime.datetime.now()
    for (i,pic) in s:
        hog = np.array(fun.HOGCalc(pic,8,9))
        sample = np.zeros((hog.size+2))
        sample[0] = i
        sample[1] = y
        sample[2:hog.size+2] = hog.reshape(hog.size)
        hog_list.append(sample)
    end_time = datetime.datetime.now()
    print((end_time-start_time).seconds)
    hog_list = np.array(hog_list)
    return  hog_list

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

def GenROC(fpos,fneg):
    fpos.sort()
    fneg.sort()
    i,j=(0,0)
    x,y=(0.,0.)
    xypoints = [[x,y]]
    while i != fpos.size-1 and j != fneg.size-1:
        if fpos[i] <= fneg[j]:
            y += 1.0 / fpos.size
            i += 1
        elif fpos[i] > fneg[j]:
            x += 1.0 / fneg.size
            j += 1
        xypoints.append([x,y])

    xypoints = np.array(xypoints)
    plt.plot(np.log10(xypoints[1:-1,0]), xypoints[1:-1,1],'r')
    plt.show()

def BuildSet(hog_pos,hog_neg,proportion=0.77):
    # Build Training set and Test set from samples s_pos and s_neg
    # Each time generate a random selection
    hog_total = np.concatenate((hog_pos,hog_neg),axis=0)
    n_total = hog_total.shape[0]
    index_train = random.sample(range(n_total),int(n_total*proportion))
    index_train.sort()
    p_train = np.zeros(n_total,dtype=np.bool)
    p_test = np.zeros(n_total,dtype=np.bool)
    for i in np.arange(n_total):
        if i in index_train :
            p_train[i] = True
        p_test[i] = not p_train[i]
    hog_train = hog_total[p_train]
    hog_test = hog_total[p_test]
    return [hog_train,hog_test]

train,test = BuildSet(hog_pos,hog_neg,0.2)

# GenROC(dist_pos,dist_neg)
def InitParams(n_param):
    return np.random.rand(n_param)

#Logistic Regression 相关函数
def LRPredict(x,w):
    return 1.0 / (1 + np.exp(-np.dot(x,w)))

def LRLoss(w,train):
    loss = 0
    for sample in train:
        #sample: [0]为帧数,[1]为类别标签,[2]~[end]为hog向量
        y = sample[1]
        x = sample[2:sample.size]
        x = np.append(x,1)
        h = LRPredict(x,w)
        loss -= y*np.log(h)+(1-y)*np.log(1-h)
    return  loss / train.shape[0]

def LRDLoss(w,train,lam=1):
    d_loss = np.zeros(train.shape[1]-1)
    for sample in train:
        y = sample[1]
        x = sample[2:sample.size]
        x = np.append(x,1)
        h = LRPredict(x,w)
        d_loss -= (y-h)*x
    return d_loss / train.shape[0] + lam * w

#LR学习主程序
def LRLearning(train):

    #参数初始化
    max_iter = 1000
    it = 0
    max_err = 1.e-4
    err = 1000
    step = 1.0
    n_param = train.shape[1]-1
    w = InitParams(n_param)
    loss = LRLoss(w,train)
    curve = []

    #主循环,采用简单学习率衰减法
    while err > max_err and it < max_iter:
        it += 1

        #计算下降方向
        grad_direct = LRDLoss(w,train)
        grad_mag = np.sqrt(sum(grad_direct**2))
        grad_direct /= -grad_mag

        #计算新参数值 w_new
        w_new = grad_direct * step
        loss_new = LRLoss(w_new,train)
        err_new = abs(loss_new - loss)
        if loss_new >= loss:
            step *= 0.8
        else:
            err,loss,w=err_new,loss_new,w_new
        print("err:%f,step:%f,loss:%f"%(err,step,loss))
        curve.append(loss)
    return w

def LRTest(w,test):
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
    plt.plot(np.log10(roc_curve[:,0]),roc_curve[:,1])
    return roc_curve


w = LRLearning(train)
LRTest(w,test)


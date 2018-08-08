from HOG.ext import functions as fun
import cv2
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
import math
import random

def softmaxLoss(w, train, k):
    loss = 0
    count = 0
    x = []
    for sample in train:
        assert isinstance(sample.size, object)
        x[count] = sample[2:sample.size]
        count += 1
    for p in range(count):
        y = train[p][1]
        z = x[p]
        for q in range(k):
            if y==q:
                loss += np.log(softmax(w, x, p, q))
    return - loss / train.shape[0]

def softmaxDLoss(w, train, j):
    d_loss = np.zeros(train.shape[1] - 1)
    count = 0
    x = []
    for sample in train:
        assert isinstance(sample.size, object)
        x[count] = sample[2:sample.size]
        count += 1
    for i in range(count):
        y = train[i][1]
        z = x[i]
        tmp = softmax(w, x, i, j)
        if y == j:
            d_loss += np.dot(z, 1-tmp)
        else:
            d_loss += np.dot(z, 0-tmp)
    return - d_loss / train.shape[0]

def softmax(w, x, i, j):
    denominator = 0
    for theta in w:
        denominator += np.exp(np.dot(x[i], theta))
    return (np.exp(np.dot(x[i], w[j]))) / denominator

def miniBatchInit(batchSize, trainingSet):
    randomNum = random.sample(range(0, trainingSet), trainingSet)
    n_batch = int(np.floor(trainingSet/batchSize))
    randomNum = np.array(randomNum)[0:n_batch*batchSize].reshape(n_batch,batchSize)
    # batchSizeNum = []
    # # for i in range(int(math.floor((trainingSet-1)/batchSize))):
    #     tmp = randomNum[i*batchSize:(i+1)*batchSize]
    #     batchSizeNum.append((tmp))
    return np.array(randomNum)

def miniBatch(image, label, batchSizeNum):
    image_out = []
    label_out = []
    for num in batchSizeNum:
        image_out.append(image[num])
        label_out.append(label[num])
    return image_out, label_out

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
    w = -InitParams(n_param)
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
        w_new = w + grad_direct * step
        loss_new = LRLoss(w_new,train)
        err_new = abs(loss_new - loss)
        if loss_new >= loss:
            step *= 0.8
        else:
            err,loss,w=err_new,loss_new,w_new
        print("err:%f,step:%f,loss:%f"%(err,step,loss))
        curve.append(loss)
    return [w,curve]

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


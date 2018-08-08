import numpy as np
import os
import struct
import  matplotlib.pyplot as plt

# def LoadMnist_all():
#     s = []
#     path = './mnist'
#     files = os.listdir(path)
#     for file in files:
#         if not os.path.isdir(file):
#             f = open(path+'/'+file)


def LoadMnistImage(path):
    f = open(path, 'rb').read()
    head = struct.unpack_from('>IIII', f, 0)

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    # [60000]*28*28
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'
    image = struct.unpack_from(bitsString, f, offset)
    image = np.reshape(image, [imgNum, width, height])
    return image


def LoadMnistLabel(path):
    f = open(path, 'rb').read()
    head = struct.unpack_from('>II', f, 0)

    offset = struct.calcsize('>II')
    imgNum = head[1]

    offset = struct.calcsize('>II')
    numString = '>'+ str(imgNum) + 'B'
    label = struct.unpack_from(numString, f, offset)
    label = np.reshape(label, [imgNum, 1])
    return label

def ShowImgLabel(i,image,label):
    plt.title("%d"%label[i])
    plt.imshow(image[i],cmap=plt.cm.gray)

def InitParams(n_label,n_param):
    w = np.random.rand(n_label,n_param)
    w[0] = 0
    return w

def PValue(w,x):
    enum = np.zeros(w.shape[0])
    for i in range(w.shape[0]):
        enum[i] = np.exp(-np.dot(w[i],x))
    denum = sum(enum)
    return enum/denum

def SftMxLoss(w,train,label):
    loss = 0
    for i in range(train.shape[0]):
        y = label[i]
        p = PValue(w,np.append(train[i],1))
        loss -= np.log(p[y])
    return loss / train.shap[0]

def SftMxDLoss(w,train,label):
    dloss = np.zeros(w.shape)
    for i in range(train.shape[0]):
        y = label[i]
        x = np.append(train[i],1)
        p = PValue(w,x)
        for j in range(1,w.shape[0]):
            id = 1 if y ==j else 0
            dloss[j] -= (id-p[j]) * x



def SftMxLearning(train,label):
    #参数初始化
    max_iter = 1000
    it = 0
    max_err = 1.e-4
    err = 1000
    step = 1.0
    n_param = train[0].size+1
    n_label = 10
    w = InitParams(n_label-1,n_param)
    loss = SftMxLoss(w,train)
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


home = 'C:/Users/peter/Documents/GitHub/SummerCamp2018/'
image = LoadMnistImage(home+'LR/mnist/train-images.idx3-ubyte')
label = LoadMnistLabel(home+'LR/mnist/train-labels.idx1-ubyte')
ShowImgLabel(100,image,label)
m = image.shape[0]
n = image[0].size
image = image.reshape(m,n)
train = np.zeros(image.shape)
for i in range(image.shape[0]):
    train[i] = image[i] / image[i].max()




import numpy as np
import  matplotlib.pyplot as plt

def ShowImgLabel(i,image,label,plabel):
    plt.title("True:%d,Predicted:%d"%(label,plabel))
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

def CalcLabel(w,x):
    p = PValue(w,x)
    return np.where(p==p.max())[0][0]


def SftMxLoss(w,train,label):
    loss = 0
    for i in range(train.shape[0]):
        y = label[i]
        x = np.append(train[i],1)
        p = PValue(w,x)
        loss -= np.log(p[y])
    return loss / train.shape[0]

def SftMxDLoss(w,train,label):
    dloss = np.zeros(w.shape)
    for i in range(train.shape[0]):
        y = label[i]
        x = np.append(train[i],1)
        p = PValue(w,x)
        for j in range(1,w.shape[0]):
            id = 1 if y ==j else 0
            dloss[j] -= (id-p[j]) * x
    return dloss / train.shape[0]



def SftMxLearning(train,label,max_iter=6000,batch_size=1000,step=10,max_err=1e-4):
    #参数初始化
    it = 0
    err = 100
    n_param = train[0].size+1
    n_label = 10
    w = InitParams(n_label,n_param)
    loss = SftMxLoss(w,train,label)
    curve = []

    #主循环,采用简单学习率衰减法
    while err > max_err and it < max_iter:
        it += 1
        batch_id = batch_size*it % train.shape[0]
        batch_train = train[batch_id:batch_id+batch_size]
        batch_label = label[batch_id:batch_id+batch_size]

        #计算下降方向
        dloss = SftMxDLoss(w,batch_train,batch_label)
        grad_mag = np.sqrt(sum(sum(dloss**2)))
        dloss /= -grad_mag

        #计算新参数值 w_new
        w_new = w - dloss * step
        loss_new = SftMxLoss(w_new,train,label)
        err_new = abs(loss_new - loss)
        if loss_new >= loss:
            step *= 0.8
        else:
            err,loss,w=err_new,loss_new,w_new
        print("iter:%d,err:%f,step:%f,loss:%f"%(it,err,step,loss))
        print("err_new:%f,loss_new:%f"%(err_new,loss_new))
        curve.append(loss)
    return [w,np.array(curve)]

def Test(w,test,test_label):
    err_count = 0
    err_record=[]
    for i in range(test.shape[0]):
        x = np.append(test[i],1)
        if CalcLabel(w,x)!=test_label[i]:
            err_record.append(i)
            err_count += 1
    return [err_record,err_count/test.shape[0]]

def CheckError(i):
    plabel=CalcLabel(w,np.append(test[i],1))
    ShowImgLabel(i,test_image,test_label[i],plabel)

home = 'C:/Users/peter/Documents/GitHub/SummerCamp2018/'
image = LoadMnistImage(home+'LR/mnist/train-images.idx3-ubyte')
label = LoadMnistLabel(home+'LR/mnist/train-labels.idx1-ubyte')
test_image = LoadMnistImage(home+'LR/mnist/t10k-images.idx3-ubyte')
test_label = LoadMnistLabel(home+'LR/mnist/t10k-labels.idx1-ubyte')
# ShowImgLabel(100,image,label)
m = image.shape[0]
n = image[0].size
image1 = image.reshape(m,n)
train = np.zeros(image1.shape)
for i in range(image1.shape[0]):
    train[i] = image1[i] / image1[i].max()

m = test_image.shape[0]
n = test_image[0].size
test_image1 = test_image.reshape(m,n)
test = np.zeros(test_image1.shape)
for i in range(test_image1.shape[0]):
    test[i] = test_image1[i] / test_image1[i].max()

[w,loss_curve] =SftMxLearning(train[0:10000],label[0:10000])
x = np.append(train[10],1)
yi = CalcLabel(w,x)
[err_record,err_rate] = Test(w,test,test_label)

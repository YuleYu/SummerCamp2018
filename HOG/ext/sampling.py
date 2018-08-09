import numpy as np
import copy
import random
#由于采用了全部计算HOG的方法,所以原先的打标签方法已经不兼容,重写打标签函数和留出法测试/训练集划分
def GenLabel(fname,nframes,winsize=1):
    #通过读取进球帧起止帧描述文件,将所有帧打标签,返回label
    f_positive = open(fname+'_pos.txt','r')
    label = np.zeros(nframes,np.uint8)
    for entry in f_positive:
        f_beg,f_end = list(map(int,entry.split()))
        for i in range(f_beg,f_end+1):
            label[i] = 1
    for i in range(nframes-winsize+1):
        if winsize == 2:
            #对2连续帧,认为包含一个进球帧即为正样本
            label[i] = 1 if label[i] or label[i+1] else 0
        elif winsize > 2:
            #对2以上连续帧,认为包含进球帧数必须不少于半数方为正样本
            label[i] = 1 if (sum(label[i:i+winsize]) >= winsize * 0.5) else 0
    f_positive.close()
    return label

def HoldOut(n,ratio_train=0.7):
    reorder = random.sample(n,n)
    n_train = int(np.floor(n*ratio_train))
    train_id = copy.deepcopy(reorder[0:n_train])
    train_id.sort()
    test_id = copy.deepcopy(reorder[n_train:n])
    test_id.sort()
    return [train_id,test_id]




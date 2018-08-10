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
            if y == q:
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
            d_loss += np.dot(z, 1 - tmp)
        else:
            d_loss += np.dot(z, 0 - tmp)
    return - d_loss / train.shape[0]


def softmax(w, x, i, j):
    denominator = 0
    for theta in w:
        denominator += np.exp(np.dot(x[i], theta))
    return (np.exp(np.dot(x[i], w[j]))) / denominator


def miniBatchInit(batchSize, trainingSet):
    randomNum = random.sample(range(0, trainingSet), trainingSet)
    n_batch = int(np.floor(trainingSet / batchSize))
    randomNum = np.array(randomNum)[0:n_batch * batchSize].reshape(n_batch, batchSize)
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


def CalcSample(s, y):
    # 计算样本的hog,正样本y=1,负样本y=0
    hog_list = []
    start_time = datetime.datetime.now()
    for (i, pic) in s:
        hog = np.array(fun.HOGCalc(pic, 8, 9))
        sample = np.zeros((hog.size + 2))
        sample[0] = i
        sample[1] = y
        sample[2:hog.size + 2] = hog.reshape(hog.size)
        hog_list.append(sample)
    end_time = datetime.datetime.now()
    print((end_time - start_time).seconds)
    hog_list = np.array(hog_list)
    return hog_list


def GenROC(fpos, fneg):
    fpos.sort()
    fneg.sort()
    i, j = (0, 0)
    x, y = (0., 0.)
    xypoints = [[x, y]]
    while i != fpos.size - 1 and j != fneg.size - 1:
        if fpos[i] <= fneg[j]:
            y += 1.0 / fpos.size
            i += 1
        elif fpos[i] > fneg[j]:
            x += 1.0 / fneg.size
            j += 1
        xypoints.append([x, y])

    xypoints = np.array(xypoints)
    plt.plot(np.log10(xypoints[1:-1, 0]), xypoints[1:-1, 1], 'r')
    plt.show()


def BuildSet(hog_pos, hog_neg, proportion=0.77):
    # Build Training set and Test set from samples s_pos and s_neg
    # Each time generate a random selection
    hog_total = np.concatenate((hog_pos, hog_neg), axis=0)
    n_total = hog_total.shape[0]
    index_train = random.sample(range(n_total), int(n_total * proportion))
    index_train.sort()
    p_train = np.zeros(n_total, dtype=np.bool)
    p_test = np.zeros(n_total, dtype=np.bool)
    for i in np.arange(n_total):
        if i in index_train:
            p_train[i] = True
        p_test[i] = not p_train[i]
    hog_train = hog_total[p_train]
    hog_test = hog_total[p_test]
    return [hog_train, hog_test]


# GenROC(dist_pos,dist_neg)
def InitParams(n_param):
    return np.random.rand(n_param)/n_param


# Logistic Regression 相关函数
def LRPredict(x, w):
    return 1.0 / (1 + np.exp(-np.dot(x, w[0:-1]) - w[-1]))


def ZeroOneLoss(w, train, label):
    y_p = np.dot(train,w[0:-1]) + w[-1] >0
    return sum(y_p != label) / train.shape[0]


def LRLoss(w, train, label, lam=0.5):
    loss = 0
    for i in range(train.shape[0]):
        y = label[i]
        x = train[i]
        h = LRPredict(x, w)
        loss -= y * np.log(h) + (1 - y) * np.log(1 - h)
    return loss / train.shape[0] + 0.5 * lam * np.dot(w, w)


def LRDLoss(w, train, label, lam=0.5):
    d_loss = np.zeros(w.shape[0])
    for i in range(train.shape[0]):
        y = label[i]
        x = train[i]
        h = LRPredict(x, w)
        d_loss[0:-1] -= (y - h) * x
        d_loss[-1] -= (y - h)
    return d_loss / train.shape[0] + lam * w


# LR学习主程序
def LRLearning(w, train, label, validate, valabel, n_epoch, batchsize, learning_rate, patience, lam, validation_frequency):
    # 仿照ppt中代码,加入minibatch,earlystop

    # 参数初始化
    epoch = 0
    n_param = train.shape[1] + 1
    # 如果val_loss没有improvement_threshold以上的提升,不改变容忍度
    improvement_threshold = 0.5
    # 如果val_loss有improvement_threshold以上提升, 将容忍度提高为迭代数的1.2倍
    patience_increase = 1.2
    n_batch = int(np.floor(train.shape[0] / batchsize))
    # w = -InitParams(n_param)
    best_validation_loss = 1
    this_validation_loss = 1
    curve = []
    done_looping = False
    batch_pool = miniBatchInit(batchsize, train.shape[0])

    #主循环
    while epoch < n_epoch and (not done_looping):
        epoch += 1
        for batch_id in range(batch_pool.shape[0]):
            minibatch_id = batch_pool[batch_id]
            batch = train[minibatch_id]
            batch_label = label[minibatch_id]
            # loss = LRLoss(w, batch, batch_label)
            dloss = LRDLoss(w, batch, batch_label, lam)
            dloss /= np.sqrt(np.dot(dloss, dloss))
            w -= dloss * learning_rate

            iter = (epoch - 1) * batch_pool.shape[0] + batch_id
            if (iter + 1) % validation_frequency == 0:
                trainloss = LRLoss(w, batch, batch_label, lam)
                print('epoch: %i, minibatch: %i/%i, iter: %i, patience: %d, training loss: %f validation error %f %%' % (
                    epoch,
                    batch_id+1,
                    batch_pool.shape[0],
                    iter,
                    patience,
                    trainloss,
                    this_validation_loss*100))
                curve.append([iter,trainloss])
                this_validation_loss = ZeroOneLoss(w, validate, valabel)
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                if patience <= iter:
                    done_looping = True
                    break
                if best_validation_loss < 1e-3:
                    break

    # #主循环,采用简单学习率衰减法
    # while err > max_err and it < max_iter:
    #     it += 1
    #
    #     #计算下降方向
    #     grad_direct = LRDLoss(w,train)
    #     grad_mag = np.sqrt(sum(grad_direct**2))
    #     grad_direct /= -grad_mag
    #
    #     #计算新参数值 w_new
    #     w_new = w + grad_direct * step
    #     loss_new = LRLoss(w_new,train)
    #     err_new = abs(loss_new - loss)
    #     if loss_new >= loss:
    #         step *= 0.8
    #     else:
    #         err,loss,w=err_new,loss_new,w_new
    #     print("err:%f,step:%f,loss:%f"%(err,step,loss))
    #     curve.append(loss)
    curve = np.array(curve)
    return [w, curve]


def LRROC(w, test):
    roc_dots = []
    for sample in test:
        y = sample[1]
        x = sample[2:sample.size]
        x = np.append(x, 1)
        value = np.dot(w, x)
        roc_dots.append([sample[0], y, value])
    roc_dots = np.array(roc_dots)
    roc_dots = roc_dots[np.lexsort(roc_dots.T)]
    (x, y) = (0, 0)
    roc_curve = [[x, y]]
    n_pos = sum(test[:, 1])
    n_neg = test.shape[0] - n_pos
    for elem in roc_dots:
        if elem[1] == 1:
            x += 1.0 / n_pos
        elif elem[1] == 0:
            y += 1.0 / n_neg
        roc_curve.append([x, y])
    roc_curve = np.array(roc_curve)
    # plt.plot(fun(roc_curve[:,0]),roc_curve[:,1])
    plt.xlim(1e-4, 1)
    plt.semilogx((roc_curve[:, 0]), roc_curve[:, 1])
    return roc_curve

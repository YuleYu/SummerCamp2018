import LR.ext.LR_Classifier as classifers
import LR.ext.Tester as tester
import numpy as np
import pickle

def train_size_fun(train_size):
    print("LR train set size:%d" % train_size)
    clsfr = classifers.LR_Classifier(w1, lam)
    x_train_new = x_train[0:train_size]
    y_train_new = y_train[0:train_size]
    w = clsfr.Training(x_train_new, y_train_new, x_test, y_test, nepoch1, batchsize, learning_rate, patience, val_freq1, 0)
    loss_curve = clsfr.loss_curve
    roc_curve = clsfr.CalROC(x_test, y_test)
    auc = clsfr.CalAUC()
    return [w, loss_curve, roc_curve, auc]

def learning_rate_fun(learning_rate):
    clsfr = classifers.LR_Classifier(w1, lam)
    w = clsfr.Training(x_train, y_train, x_test, y_test, nepoch1, batchsize, learning_rate, patience, val_freq1, 0)
    loss_curve = clsfr.loss_curve
    roc_curve = clsfr.CalROC(x_test, y_test)
    auc = clsfr.CalAUC()
    return [w, loss_curve, roc_curve, auc]

def batchfun(size):
    clsfr = classifers.LR_Classifier(w1, lam)
    w = clsfr.Training(x_train, y_train, x_test, y_test, nepoch1, size, learning_rate, patience, val_freq1, 0)
    loss_curve = clsfr.loss_curve
    roc_curve = clsfr.CalROC(x_test, y_test)
    auc = clsfr.CalAUC()
    clsfr.ReRoll()
    return [w, loss_curve, roc_curve, auc]

def methodfun(method):
    clsfr = classifers.LR_Classifier(w1, lam)
    clsfr.ReRoll()
    w = clsfr.Training(x_train, y_train, x_test, y_test, epochs[method], batchsize, learning_rate, patience, val_freqs[method],
                       method)
    loss_curve = clsfr.loss_curve
    roc_curve = clsfr.CalROC(x_test, y_test)
    auc = clsfr.CalAUC()
    return [w, loss_curve, roc_curve, auc]


# 此文件为Logistic Regression Classifier 主函数,
# 将原先的mnist相关代码转移到了LR/main_MNIST中

home = 'C:/Users/peter/Documents/GitHub/SummerCamp2018/'

f_train = open(home+'HOG/result/train.dat','rb')
f_test = open(home+'HOG/result/test.dat','rb')

x_train=pickle.load(f_train)
y_train=pickle.load(f_train)
f_train.close()

x_test=pickle.load(f_test)
y_test=pickle.load(f_test)
f_test.close()


# 进行LR分类器训练的初始化
learning_rate = 0.4
patience = 1000000
nepoch1 = 100
batchsize = 200
lam = 1e-6
val_freq1 = 50
w1 = np.zeros(x_test.shape[1] + 1)

print('Which test?[train size,learning rate,batch size, method,all]')
choice = input()


if 'train' in choice:
    # 训练集大小对分类效果的影响
    train_size_list = [1000, 2000, 4000, 8000, 16000, 32000, 45000]
    train_size_tester = tester.Classifier_Tester('LR','Train set size', train_size_list, train_size_fun)
    train_size_tester.Running()
    train_size_tester.PlotROC('Train set size')
    train_size_tester.PlotLoss('Train set size')
elif 'learning' in choice:
    #学习率影响
    learning_rates = np.arange(0.1, 0.4, 0.1)
    learning_rate_tester = tester.Classifier_Tester('LR','Learning rate', learning_rates, learning_rate_fun)
    learning_rate_tester.Running()
    learning_rate_tester.PlotROC('Learning_rate')
    learning_rate_tester.PlotLoss('Learning_rate')
elif 'batch' in choice:
    # 不同batchsize
    batchsizes = np.arange(100, 600, 100)
    batch_tests = tester.Classifier_Tester('LR','BatchSize', batchsizes, batchfun)
    batch_tests.Running()
    batch_tests.PlotROC('batch_size')
    batch_tests.PlotLoss('batch_size')
elif 'method' in choice:
    # 不同method
    val_freqs = [62, 62, 1]
    methods = [0, 2]
    epochs = [100,100,20]
    method_test = tester.Classifier_Tester('LR','Method', methods, methodfun, ['Grad Descent', 'Newton'])
    method_test.Running()
    method_test.PlotROC('method')
    method_test.PlotLoss('method')


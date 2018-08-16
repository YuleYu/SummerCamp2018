import HOG.ext.hog_compute as hg_cpt
import HOG.ext.sampling as sampling
import HOG.ext.ReadXML
import LR.ext.Functions as lr
import LR.ext.LR_Classifier as classifers
import LR.ext.Tester as tester
import numpy as np
import matplotlib.pyplot as plt
import itertools

# 此文件为Logistic Regression Classifier 主函数,
# 将原先的mnist相关代码转移到了LR/main_MNIST中

home = 'C:/Users/peter/Documents/GitHub/SummerCamp2018/'
videochoice = "video1"
fname = home + 'HOG/result/' + videochoice
imgPath, size_x, size_y, startTime, endTime, colSize, fps, bin = HOG.ext.ReadXML.ReadXML(
    home + 'HOG/script/' + videochoice + '.xml')
n_frames = (endTime - startTime) * fps

# 加载HOG,并用HoldOut留出法划分测试集训练集
h = hg_cpt.LoadHOG(fname)
label = sampling.GenLabel(fname, n_frames)

# 对双帧样本进行留出法划分训练集
# x1 = h
# x2 = np.zeros((x1.shape[0] - 1, x1.shape[1] * 2))
# x2[:, 0:x1.shape[1]] = x1[0:-1]
# x2[:, x1.shape[1]:2 * x1.shape[1]] = x1[1:x1.shape[0]]
# y2 = sampling.GenLabel(fname, n_frames, winsize=2)[0:-1]
# train_id2, test_id2 = sampling.HoldOut(n_frames - 1)
# x_train2 = x2[train_id2]
# y_train2 = y2[train_id2]
# x_test2 = x2[test_id2]
# y_test2 = y2[test_id2]

# 进行LR分类器训练的初始化
learning_rate = 0.4
patience = 1000000
nepoch1 = 100
batchsize = 200
lam = 1e-7

# 训练集大小对分类效果的影响
train_size_list = [1000, 2000, 4000, 8000, 16000, 32000, 45000]
train_id, test_id = sampling.HoldOut(h.shape[0])
n_pos = sum(label)
pos_id = label == 1
w1 = np.zeros(h.shape[1] + 1)
lr_train_size = classifers.LR_Classifier(w1, lam)
result_list = []
for size_train in train_size_list:
    x_train = h[train_id][0:size_train]
    y_train = label[train_id][0:size_train]
    x_test = h[test_id]
    y_test = label[test_id]
    val_freq1 = int(size_train / batchsize)

    print("LR train set size:%d" % size_train)
    lr_train_size.Training(x_train, y_train, x_test, y_test, nepoch1, batchsize, learning_rate, patience, val_freq1, 0)
    loss_curve = lr_train_size.loss_curve
    roc_curve = lr_train_size.CalROC(x_test, y_test)
    auc = lr_train_size.CalAUC()
    lr_train_size.ReRoll()

    result_list.append([w1, loss_curve, roc_curve, auc])

legend_list = list(
    'Train set size:%d, AUC:%f' % (train_size_list[i], result_list[i][3]) for i in range(len(train_size_list)))
style_list = list(
    '%s%s-' % (style[1], style[0]) for style in itertools.product(['*', '+', 'v'], ['b', 'r', 'y', 'm', 'c', 'g']))
fig, ax1 = plt.subplots()
ax1.grid(True)
ax1.set_title('ROC of LR with Different Data Size')
ax1.set_xlabel('False Alarm Rate')
ax1.set_ylabel('Recall')
plt.xlim(1e-5, 1)
plt.ylim(0, 1)
plot_list = []
for i in range(len(result_list)):
    roc_curve1 = result_list[i][2]
    l1, = ax1.semilogx(roc_curve1[:, 0], roc_curve1[:, 1], style_list[i], markevery=20)
    plot_list.append(l1)
ax1.legend(plot_list, legend_list, loc='lower right')  # 其中，loc表示位置的；
plt.savefig("train_set_size.svg", format="svg")

fig, ax1 = plt.subplots()
ax1.grid(True)
ax1.set_title('Loss of LR with Different Data Size')
ax1.set_xlabel('iter')
ax1.set_ylabel('loss')
plot_list = []
for i in range(len(result_list)):
    loss_curve1 = result_list[i][1]
    l1, = ax1.plot(loss_curve1[:, 0], loss_curve1[:, 1], style_list[i], markevery=1)
    plot_list.append(l1)
ax1.legend(plot_list, legend_list, loc='lower right')  # 其中，loc表示位置的；
plt.savefig("loss_train_set_size.svg", format="svg")

# 初始化train,test,后面统一
train_id, test_id = sampling.HoldOut(h.shape[0])
x_train = h[train_id]
y_train = label[train_id]
x_test = h[test_id]
y_test = label[test_id]

# 不同学习率影响
patience = 10000
val_freq1 = 50
batchsize = 300
nepoch1 = 100
lam = 1e-7
learning_rates = np.arange(0.1, 0.4, 0.1)
w1 = np.zeros(x_test.shape[1] + 1)


def learning_rate_fun(learning_rate):
    clsfr = classifers.LR_Classifier(w1, lam)
    w = clsfr.Training(x_train, y_train, x_test, y_test, nepoch1, batchsize, learning_rate, patience, val_freq1, 0)
    loss_curve = clsfr.loss_curve
    roc_curve = clsfr.CalROC(x_test, y_test)
    auc = clsfr.CalAUC()
    return [w, loss_curve, roc_curve, auc]


learning_rate_tester = tester.Classifier_Tester('LR','Learning rate', learning_rates, learning_rate_fun)
learning_rate_tester.Running()
learning_rate_tester.PlotROC('Learning_rates')
learning_rate_tester.PlotLoss('Learning_rates')

# 不同batchsize
lam1 = 1e-7
learning_rate = 0.4
result_list3 = []
batchsizes = np.arange(100, 600, 100)


def batchfun(size):
    clsfr = classifers.LR_Classifier(w1, lam)
    w = clsfr.Training(x_train, y_train, x_test, y_test, nepoch1, size, learning_rate, patience, val_freq1, 0)
    loss_curve = clsfr.loss_curve
    roc_curve = clsfr.CalROC(x_test, y_test)
    auc = clsfr.CalAUC()
    clsfr.ReRoll()
    return [w, loss_curve, roc_curve, auc]


batch_tests = tester.Classifier_Tester('LR','BatchSize', batchsizes, batchfun)
batch_tests.Running()
batch_tests.PlotROC('batch_size')
batch_tests.PlotLoss('batch_size')

# 不同method
patience = 30000
val_freqs = [62, 62, 1]
nepoch1 = 10
lam = 1e-6
learning_rate = 0.4
batchsize = 500
result_list3 = []
methods = [0, 2]


def methodfun(method):
    clsfr = classifers.LR_Classifier(w1, lam)
    clsfr.ReRoll()
    w = clsfr.Training(x_train, y_train, x_test, y_test, nepoch1, batchsize, learning_rate, patience, val_freqs[method],
                       method)
    loss_curve = clsfr.loss_curve
    roc_curve = clsfr.CalROC(x_test, y_test)
    auc = clsfr.CalAUC()
    return [w, loss_curve, roc_curve, auc]


method_test = tester.Classifier_Tester('LR','Method', methods, methodfun, ['Grad Descent', 'Newton'])
method_test.Running()
method_test.PlotROC('method')
method_test.PlotLoss('method')


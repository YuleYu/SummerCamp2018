import LR.ext.Functions as lr
import svmutil
import HOG.ext.hog_compute as hg_cpt
import HOG.ext.sampling as sampling
import HOG.ext.ReadXML
import numpy as np
import matplotlib.pyplot as plt
import profile

home = 'C:/Users/peter/Documents/GitHub/SummerCamp2018/'
videochoice = "video1"
fname = home+'HOG/result/'+videochoice
imgPath, size_x, size_y, startTime, endTime, colSize, fps, bin = HOG.ext.ReadXML.ReadXML(home+'HOG/script/'+videochoice+'.xml')
n_frames = (endTime - startTime) * fps

# 加载HOG,并用HoldOut留出法划分测试集训练集
h = hg_cpt.LoadHOG(fname)
label = sampling.GenLabel(fname,n_frames)
train_id,test_id = sampling.HoldOut(label.shape[0])
x_train = h[train_id]
y_train = label[train_id]
x_test = h[test_id]
y_test = label[test_id]

y_svm = list(y_train)
id = np.arange(h.shape[1])
x_svm = [dict(zip(np.arange(x_train.shape[1]),x_train[i])) for i in np.arange(x_train.shape[0])]
profile.run('x_svm = list(dict(zip(np.arange(x_train.shape[1]),x_train[i])) for i in np.arange(x_train.shape[0]))')
m = svmutil.svm_train(y_svm,x_svm,'-c 0.5 -t 2 -g 1 -b 1')
profile.run("m = svmutil.svm_train(y_svm,x_svm,'-c 0.5 -t 2 -g 1 -b 1')")
svmutil.svm_save_model('c_05_gamma_1.model',m)
y_svm_test = list(y_test)
x_svm_test = list(dict(zip(np.arange(x_test.shape[1]),x_test[i])) for i in np.arange(x_test.shape[0]))

pre_label, acc, dec = svmutil.svm_predict(y_svm_test,x_svm_test,m, '-b 1')
dec_ar = np.array(dec)
order = np.lexsort([dec_ar[:,1]])
x_p,y_p = (1,1)
n_pos = sum(y_test)
n_neg = y_test.shape[0] - n_pos
curve1 = []
for i in order:
    if y_test[i] == 1:
        y_p -= 1 / n_pos
    else:
        x_p -= 1 / n_neg
    curve1.append([x_p,y_p])
curve1 = np.array(curve1)
# plt.plot(curve1[:,0],curve1[:,1])

patience = 30000
val_freq1 = 62
nepoch1 = 100
lam1 = 1e-6
learning_rate = 0.4
batchsize = 500

w1 = np.zeros(x_train.shape[1]+1)
[w1,curve]=lr.LRLearning(w1, x_train, y_train, x_test, y_test, 20, batchsize, learning_rate, patience, lam1, 1,2)
roc_curve= lr.LRROC(w1,x_test,y_test )

fig1,ax1 = plt.subplots()
ax1.grid(True)
ax1.set_title('ROC of a SVM Classifier')
ax1.set_xlabel('False Alarm Rate')
ax1.set_ylabel('Recall')
plt.xlim(1e-5,1)
l1,=ax1.semilogx(curve1[:,0],curve1[:,1],'r*-',markevery=20)
l2,=ax1.semilogx(roc_curve[:,0],roc_curve[:,1],'gh-',markevery=20)
ax1.legend([l1,l2], ['SVM_C=0.5$\gamma=$1','LR with Newton Method'], loc = 'lower right')             #其中，loc表示位置的；
plt.savefig('svm.svg',format='svg')

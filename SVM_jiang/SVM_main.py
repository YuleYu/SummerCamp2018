import LR.ext.Functions as lr
import svmutil
import HOG.ext.hog_compute as hg_cpt
import HOG.ext.sampling as sampling
import HOG.ext.ReadXML
import numpy as np
import matplotlib.pyplot as plt
import os
import profile
import pickle
import SVM_jiang.ext.SVM_Classifier as classifier
import LR.ext.Tester as tester

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
x_svm = [dict(zip(np.arange(x_train.shape[1]),x_train[i])) for i in np.arange(x_train.shape[0])]
y_svm_test = list(y_test)
x_svm_test = list(dict(zip(np.arange(x_test.shape[1]),x_test[i])) for i in np.arange(x_test.shape[0]))

#对C测试
def c_value_fun(c):
    fmodel = 'c_{c:.2f}_g_{g:.2f}.model'.format(c=c,g=1)
    if os.path.exists(fname):
        m = svmutil.svm_load_model(fmodel)
    else:
        opt_str = '-c {c} -t 2 -g {g} -b 1'.format(c = c, g = 1)
        m = svmutil.svm_train(y_svm,x_svm,opt_str)
    clsfr = classifier.SVM_Classifier(m)
    clsfr.SaveModel(fmodel)
    roc_curve=clsfr.CalROC(x_svm_test,y_svm_test)
    auc=clsfr.CalAUC()
    return [m,[],roc_curve,auc]

c_values = 2.**np.arange(-2,6,2)
c_tester = tester.Classifier_Tester('SVM','C',c_values,c_value_fun)
c_tester.Running()
c_tester.PlotROC('SVM_C')


#对Gamma测试
def g_value_fun(g):
    fmodel = 'c_{c:.2f}_g_{g:.2f}.model'.format(c = 4,g = g)
    if os.path.exists(fname):
        m = svmutil.svm_load_model(fmodel)
    else:
        opt_str = '-c {c} -t 2 -g {g} -b 1'.format(c = 4, g = g)
        m = svmutil.svm_train(y_svm,x_svm,opt_str)
        svmutil.svm_save_model(fmodel,m)
    clsfr = classifier.SVM_Classifier(m)
    roc_curve=clsfr.CalROC(x_svm_test,y_svm_test)
    auc=clsfr.CalAUC()
    return [m,[],roc_curve,auc]

g_values = 10.**np.arange(-3,1,1)
g_tester = tester.Classifier_Tester('SVM','Gamma',g_values,g_value_fun)
g_tester.Running()
g_tester.PlotROC('SVM_Gamma')


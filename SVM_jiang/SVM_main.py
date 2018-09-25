import svmutil
import numpy as np
import os
import SVM_jiang.ext.SVM_Classifier as classifier
import LR.ext.Tester as tester
import pickle

home = 'C:/Users/peter/Documents/GitHub/SummerCamp2018/'

f_train = open(home+'HOG/result/train.dat','rb')
f_test = open(home+'HOG/result/test.dat','rb')

x_train=pickle.load(f_train)
y_train=pickle.load(f_train)
f_train.close()

x_test=pickle.load(f_test)
y_test=pickle.load(f_test)
f_test.close()
y_svm = list(y_train)
x_svm = [dict(zip(np.arange(x_train.shape[1]),x_train[i])) for i in np.arange(x_train.shape[0])]
y_svm_test = list(y_test)
x_svm_test = list(dict(zip(np.arange(x_test.shape[1]),x_test[i])) for i in np.arange(x_test.shape[0]))

#对C测试
def c_value_fun(c):
    fmodel = 'c_{c:.2f}_g_{g:.3f}.model'.format(c=c,g=1)
    if os.path.exists(fmodel):
        print("model %s found, loading ..." % fmodel)
        m = svmutil.svm_load_model(fmodel)
        print("model %s loaded" % fmodel)
    else:
        print("model %s not found, training ..." %fmodel)
        opt_str = '-c {c} -t 2 -g {g} -b 1'.format(c = c, g = 1)
        m = svmutil.svm_train(y_svm,x_svm,opt_str)
        print("saving model %s ..." % fmodel)
        svmutil.svm_save_model(fmodel,m)
    clsfr = classifier.SVM_Classifier(m)
    roc_curve=clsfr.CalROC(x_svm_test,y_svm_test)
    auc=clsfr.CalAUC()
    return [m,[],roc_curve,auc]

# c_values = 2.**np.arange(-2,6,2)
# c_tester = tester.Classifier_Tester('SVM','C',c_values,c_value_fun)
# c_tester.Running()
# c_tester.PlotROC('SVM_C')


#对Gamma测试
def g_value_fun(g):
    fmodel = 'c_{c:.2f}_g_{g:.3f}.model'.format(c = 0.25,g = g)
    if os.path.exists(fmodel):
        print("model %s found, loading ..." % fmodel)
        m = svmutil.svm_load_model(fmodel)
        print("model %s loaded" % fmodel)
    else:
        print("model %s not found, training ..." %fmodel)
        opt_str = '-c {c} -t 2 -g {g} -b 1'.format(c = 0.25, g = g)
        m = svmutil.svm_train(y_svm,x_svm,opt_str)
        print("saving model %s ..." % fmodel)
        svmutil.svm_save_model(fmodel,m)
    clsfr = classifier.SVM_Classifier(m)
    roc_curve=clsfr.CalROC(x_svm_test,y_svm_test)
    auc=clsfr.CalAUC()
    return [m,[],roc_curve,auc]

g_values = 10.**np.arange(-3,1,1)
g_tester = tester.Classifier_Tester('SVM','Gamma',g_values,g_value_fun)
g_tester.Running()
g_tester.PlotROC('SVM_Gamma')


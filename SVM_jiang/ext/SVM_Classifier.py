import numpy as np
from LR.ext.Classifiers import Classifier
import svmutil

class SVM_Classifier(Classifier):

    __name__ = 'SVM'
    def __init__(self, m):
        self.m = m
        self.roc_curve = np.array([])
        self.auc = 0.

    def CalROC(self, x_val, y_val):
        prelabel,acc,dec = svmutil.svm_predict(y_val,x_val,self.m, '-b 1')
        dec_ar = np.array(dec)
        order = np.lexsort([dec_ar[:,1]])
        (x, y) = (1, 1)
        roc_curve = [[x,y]]
        n_pos = sum(y_val)
        n_neg = len(y_val) - n_pos
        for i in order:
            if y_val[i] == 1:
                y -= 1.0 / n_pos
            elif y_val[i] == 0:
                x -= 1.0 / n_neg
            roc_curve.append([x, y])
        self.roc_curve = np.array(roc_curve)
        return self.roc_curve

    def CalAUC(self):
        self.auc = 0
        for i in range(self.roc_curve.shape[0]-1):
            x_1, y_1 = self.roc_curve[i]
            x_2, y_2 = self.roc_curve[i+1]
            self.auc += np.abs(x_2 - x_1) * 0.5 * (y_2 + y_1)
        return self.auc

    def ClearModel(self):
        self.roc_curve = np.array([])
        self.auc = 0

    def SaveModel(self,fname):
        svmutil.svm_save_model(fname,self.m)

    def LoadModel(self,fname):
        self.ClearModel()
        self.m = svmutil.svm_load_model(fname)



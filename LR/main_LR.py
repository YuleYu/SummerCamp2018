import HOG.ext.hog_compute as hg_cpt
import HOG.ext.sampling as sampling
import HOG.ext.ReadXML
import LR.ext.LoadMnist
from LR.ext.Functions import *
import numpy as np

#此文件为Logistic Regression Classifier 主函数,
# 将原先的mnist相关代码转移到了LR/main_MNIST中

home = 'C:/Users/peter/Documents/GitHub/SummerCamp2018/'
videochoice = "video1"
fname = home+'HOG/result/'+videochoice
imgPath, size_x, size_y, startTime, endTime, colSize, fps, bin = HOG.ext.ReadXML.ReadXML(home+'HOG/script/'+videochoice+'.xml')
n_frames = (endTime - startTime) * fps

# 加载HOG,并用HoldOut留出法划分测试集训练集
h = hg_cpt.LoadHOG(fname)
label = sampling.GenLabel(fname,n_frames)
train_id, test_id = sampling.HoldOut(n_frames)
x_train = h[train_id]
y_train = label[train_id]
x_test = h[test_id]
y_test = label[test_id]

#进行LR分类器训练
# w = LRLearning(x_train,y_train,x_test,y_test)



import LR.ext.LR_Classifier as classifers
import NN.ext.MLP_Classifier as mlp
import LR.ext.Tester as tester
import numpy as np
import pickle
import matplotlib.pyplot as plt

home = 'C:/Users/peter/Documents/GitHub/SummerCamp2018/'

f_train = open(home+'HOG/result/train.dat','rb')
f_test = open(home+'HOG/result/test.dat','rb')

x_train=pickle.load(f_train)
y_train=pickle.load(f_train)
f_train.close()

x_test=pickle.load(f_test)
y_test=pickle.load(f_test)
f_test.close()

tanhfun = mlp.Active_Fun(np.tanh,#激活函数
                         (lambda a:1 - a ** 2),#激活函数的导数
                         (lambda n_i,n_o: 4.0*np.sqrt(6.0/(n_i+n_o)))) #激活函数对应的初始化函数
softmaxfun = mlp.Active_Fun((lambda z: (np.exp(z.T)/(np.sum(np.exp(z),axis=1))).T),
                            (lambda a:1 - a),
                            (lambda n_i,n_o: np.sqrt(6.0/(n_i + n_o))))

learning_rate = 0.1
learning_rate *= 0.618
learning_rate /= 0.618
L2_reg = 1e-2
new_nn = mlp.MLP_Classifier([900,10,2],[tanhfun,softmaxfun])
new_nn.Training(x_train,y_train,x_test,y_test,50,100,learning_rate,100000,200,0)
curve = new_nn.loss_curve
plt.plot(curve[:,0],curve[:,1])
roc = new_nn.calroc(x_test,y_test)
plt.xlim(1e-5,1e-1)
plt.semilogx(roc[:,0],roc[:,1])
plt.plot(roc[:,0],roc[:,1])
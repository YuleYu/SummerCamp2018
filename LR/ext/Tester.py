import numpy as np
import itertools
import matplotlib.pyplot as plt

class Classifier_Tester:
    name = ''
    values = np.array([])
    fun = lambda x : x
    results = []
    def __init__(self,testname,varname,values,fun,legends=[]):
        self.testname = testname
        self.varname = varname
        self.values = values
        self.fun = fun
        self.results = []
        if legends ==[]:
            self.legend_template = '{name}:{val:.2f}, AUC:{auc:.4f}'
            self.legends = self.values
        else:
            self.legend_template = '{name}:{val}, AUC:{auc:.4f}'
            self.legends = legends

    def Running(self):
        for val in self.values:
            print('Testing {name}:{val}'.format(name=self.name,val=val))
            self.results.append(self.fun(val))
    def PlotROC(self,fname):
        legend_list = list(self.legend_template.format(name=self.name,val=self.legends[i],auc=self.results[i][3]) for i in range(len(self.values)))
        style_list = list('%s%s-'%(style[1],style[0]) for style in itertools.product(['*','+','v'],['b','r','y','m','c','g']))
        fig,ax = plt.subplots()
        ax.grid(True)
        ax.set_title('ROC of {testname} with Different {name}'.format(testname=self.testname, name=self.name))
        ax.set_xlabel('False Alarm Rate')
        ax.set_ylabel('Recall')
        plt.xlim(1e-5,1e-1)
        plt.ylim(0,1)
        plot_list = []
        for i in range(len(self.values)):
            roc_curve1 = self.results[i][2]
            l, = ax.semilogx(roc_curve1[:,0],roc_curve1[:,1],style_list[i], markevery=20)
            plot_list.append(l)
        ax.legend(plot_list, legend_list, loc = 'lower right')             #其中，loc表示位置的；
        plt.savefig(fname+'_roc.svg', format="svg")


    def PlotLoss(self,fname):
        legend_list = list(self.legend_template.format(name=self.name,val=self.values[i],auc=self.results[i][3]) for i in range(len(self.values)))
        style_list = list('%s%s-'%(style[1],style[0]) for style in itertools.product(['*','+','v'],['b','r','y','m','c','g']))
        fig,ax = plt.subplots()
        ax.grid(True)
        ax.set_title('Loss of {testname} with Different {name}'.format(testname=self.testname,name=self.name))
        ax.set_xlabel('iter')
        ax.set_ylabel('loss')
        plot_list = []
        for i in range(len(self.values)):
            loss_curve = self.results[i][1]
            l, = ax.plot(loss_curve[:,0],loss_curve[:,1],style_list[i], markevery=20)
            plot_list.append(l)
        ax.legend(plot_list, legend_list, loc = 'lower right')             #其中，loc表示位置的；
        plt.savefig(fname+'_loss.svg', format="svg")




import HOG.ext.sampling as sampling
import HOG.ext.hog_compute as hog_compute
import numpy as np
import matplotlib.pyplot as plt

home = './'
fname = home + 'result/video1'
nframe = 30 * 60 * 25
x1 = hog_compute.LoadHOG(fname)
x2 = np.zeros((x1.shape[0]-1,x1.shape[1]*2))
x2[:,0:x1.shape[1]]=x1[0:-1]
x2[:,x1.shape[1]:2*x1.shape[1]]=x1[1:x1.shape[0]]
y1 = sampling.GenLabel(fname, nframe, winsize=1)
y2 = sampling.GenLabel(fname, nframe, winsize=2)[0:-1]

center_1 = sum(x1[y1.astype(np.bool)])/x1.shape[0]
center_2 = sum(x2[y2.astype(np.bool)])/(x2.shape[0])


dist1 = np.array(list(map(lambda x:np.sqrt(np.dot(x,x)), x1-center_1)))
dist2 = np.array(list(map(lambda x:np.sqrt(np.dot(x,x)), x2-center_2)))

order1 = np.lexsort([dist1])
order2 = np.lexsort([dist2])

x,y = (1,1)
n_pos = sum(y1)
n_neg = y1.shape[0] - n_pos
curve1 = []
for i in order1:
    if y1[i] == 1:
        y -= 1 / n_pos
    else:
        x -= 1 / n_neg
    curve1.append([x,y])
curve1 = np.array(curve1)

fig1,ax1 = plt.subplots()
ax1.grid(True)
ax1.set_title('ROC of Naive distance classifier')
ax1.set_xlabel('False Alarm Rate')
ax1.set_ylabel('Recall')
plt.xlim(1e-5,1)

l1,=ax1.semilogx(curve1[:,0],curve1[:,1],'gh-',markevery=100)

x,y = (1,1)
n_pos = sum(y2)
n_neg = y2.shape[0] - n_pos
curve2 = []
for i in order2:
    if y2[i] == 1:
        y -= 1 / n_pos
    else:
        x -= 1 / n_neg
    curve2.append([x,y])
curve2 = np.array(curve2)
l2,=ax1.semilogx(curve2[:,0],curve2[:,1],'r*-',markevery=100)
ax1.legend([l1,l2], ['Single Frame','Double Frame'], loc = 'lower right')
plt.savefig('roc_naive.svg',format='svg')

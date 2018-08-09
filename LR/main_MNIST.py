import HOG.ext.test
import LR.ext.LoadMnist
import numpy as np

path = []
path.append('./mnist/t10k-images.idx3-ubyte')   # path[0]
path.append('./mnist/t10k-labels.idx1-ubyte')   # path[1]
path.append('./mnist/train-images.idx3-ubyte')  # path[2]
path.append('./mnist/train-labels.idx1-ubyte')  # path[3]

train_image = LR.ext.LoadMnist.LoadMnistImage(path[2])
train_label = LR.ext.LoadMnist.LoadMnistLabel(path[3])
test_image = LR.ext.LoadMnist.LoadMnistImage(path[0])
test_label = LR.ext.LoadMnist.LoadMnistLabel(path[1])

train_image = np.reshape(train_image, [train_image.shape[0], train_image.shape[1]*train_image.shape[2]])
train_data = np.append(train_label, train_image, axis=1)

test_image = np.reshape(test_image, [test_image.shape[0], test_image[1]*test_image[2]])
test_data = np.append(test_label, test_image, axis=1)

w = HOG.ext.test.LRLearning(train_image)
HOG.ext.test.LRTest(w, test_image)

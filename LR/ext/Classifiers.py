import numpy as np
import random
from scipy.linalg import solve

class Classifier:

    __name__ = ''
    loss_curve = np.array([])

    def miniBatchInit(self, batchSize, trainingSet):
        randomNum = random.sample(range(0, trainingSet), trainingSet)
        n_batch = int(np.floor(trainingSet / batchSize))
        randomNum = np.array(randomNum)[0:n_batch * batchSize].reshape(n_batch, batchSize)
        return np.array(randomNum)


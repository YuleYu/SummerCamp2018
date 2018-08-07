import numpy as np

def softmaxLoss(w, train, k):
    loss = 0
    count = 0
    x = []
    for sample in train:
        assert isinstance(sample.size, object)
        x[count] = sample[2:sample.size]
        count += 1
    for p in range(count):
        y = train[p][1]
        z = x[p]
        for q in range(k):
            if y==q:
                loss += np.log(softmax(w, x, p, q))
    return - loss / train.shape[0]

def softmaxDLoss(w, train, j):
    d_loss = np.zeros(train.shape[1] - 1)
    count = 0
    x = []
    for sample in train:
        assert isinstance(sample.size, object)
        x[count] = sample[2:sample.size]
        count += 1
    for i in range(count):
        y = train[i][1]
        z = x[i]
        tmp = softmax(w, x, i, j)
        if y == j:
            d_loss += np.dot(z, 1-tmp)
        else:
            d_loss += np.dot(z, 0-tmp)
    return - d_loss / train.shape[0]

def softmax(w, x, i, j):
    denominator = 0
    for theta in w:
        denominator += np.exp(np.dot(x[i], theta))
    return (np.exp(np.dot(x[i], w[j]))) / denominator
import numpy as np
from scipy.linalg import solve
from LR.ext.Classifiers import Classifier

class LR_Classifier(Classifier):

    __name__ = 'LR'
    def __init__(self, w, lam):
        self.w = w
        self.w_best = w
        self.lam = lam
        self.epoch = 0
        self.roc_curve = np.array([])
        self.auc = 0.

    def ZeroOneLoss(self, xs, ys):
        y_p = np.dot(xs, self.w[0:-1]) + self.w[-1] > 0
        return sum(y_p != ys) / xs.shape[0]

    def Loss(self, xs, ys):
        h = 1.0 / (1 + np.exp(-np.dot(xs, self.w[0:-1]) - self.w[-1]))
        return -sum(ys * np.log(h) + (1 - ys) * np.log(1 - h)) / xs.shape[0] + 0.5 * self.lam * np.dot(self.w, self.w)

    def DLoss(self, xs, ys):
        d_loss = np.zeros(self.w.shape[0])
        h = 1.0 / (1 + np.exp((-np.dot(xs, self.w[0:-1]) - self.w[-1])))
        d_loss[0:-1] -= np.dot(xs.T, ys - h)
        d_loss[-1] -= sum(ys - h)
        return d_loss / xs.shape[0] + self.lam * self.w

    def DDLoss(self, xs):
        dd_loss = np.zeros((self.w.shape[0], self.w.shape[0]))
        h = 1.0 / (1 + np.exp((-np.dot(xs, self.w[0:-1]) - self.w[-1])))
        train_h = np.transpose([h]) * xs
        h1 = 1-h
        train_h1 = xs - train_h
        dd_loss[0:-1,0:-1]= np.dot(train_h.T,train_h1)
        dd_loss[-1,0:-1] = np.dot(train_h.T,h1)
        dd_loss[0:-1,-1] = np.dot(train_h.T,h1)
        dd_loss[-1,-1] = np.dot(h,h1)
        return dd_loss / xs.shape[0] + self.lam * np.identity(self.w.shape[0])

    def Training(self, x_train, y_train, x_val, y_val, n_epoch, batchsize, learning_rate, patience,
                 validation_frequency, search_method, obj_val_loss=3.7e-3):
        # 参数初始化
        # 如果val_loss没有improvement_threshold以上的提升,不改变容忍度
        improvement_threshold = 1
        # 如果val_loss有improvement_threshold以上提升, 将容忍度提高为迭代数的1.2倍
        patience_increase = 1.5
        best_validation_loss = self.ZeroOneLoss(x_val, y_val)
        this_validation_loss = best_validation_loss
        done_looping = False
        curve = []

        # 主循环
        while self.epoch < n_epoch and (not done_looping):
            self.epoch += 1
            batch_pool = self.miniBatchInit(batchsize, x_train.shape[0])
            if search_method == 0: # Gradient without line search
                for batch_id in range(batch_pool.shape[0]):
                    minibatch_id = batch_pool[batch_id]
                    batch = x_train[minibatch_id]
                    batch_label = y_train[minibatch_id]
                    dloss = self.DLoss(batch, batch_label)
                    dloss /= np.sqrt(np.dot(dloss, dloss))
                    self.w -= dloss * learning_rate

                    iter = (self.epoch - 1) * batch_pool.shape[0] + batch_id
                    if (iter + 1) % validation_frequency == 0:
                        trainloss = self.Loss(x_train, y_train)
                        print( 'epoch: %i, minibatch: %i/%i, iter: %i, patience: %d, training loss: %f validation error %f %% best validation error %f %%' % (
                            self.epoch,
                            batch_id + 1,
                            batch_pool.shape[0],
                            iter,
                            patience,
                            trainloss,
                            this_validation_loss * 100,
                            best_validation_loss * 100))
                        curve.append([iter, trainloss])
                        this_validation_loss = self.ZeroOneLoss(x_val, y_val)
                        if this_validation_loss < best_validation_loss:
                            if this_validation_loss < best_validation_loss * improvement_threshold:
                                patience = max(patience, iter * patience_increase)
                            best_validation_loss = this_validation_loss
                            self.w_best = self.w
                        if patience <= iter:
                            done_looping = True
                            break
                        if best_validation_loss < obj_val_loss:
                            break

            elif search_method == 2: #Newton method 牛顿法
                dloss = self.DLoss(x_train, y_train)
                ddloss = self.DDLoss(x_train)
                self.w -= solve(ddloss,dloss)
                iter = self.epoch - 1
                if (iter + 1) % validation_frequency == 0:
                    trainloss = self.Loss(x_train, y_train)
                    print('epoch: %i, iter: %i, patience: %d, training loss: %f validation error %f %%, best val_error: %f %%' % (
                        self.epoch,
                        iter,
                        patience,
                        trainloss,
                        this_validation_loss * 100,
                        best_validation_loss * 100))
                    curve.append([iter, trainloss])
                    this_validation_loss = self.ZeroOneLoss(x_val, y_val)
                    if this_validation_loss < best_validation_loss:
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iter * patience_increase)
                        best_validation_loss = this_validation_loss
                        self.w_best = self.w
                    if patience <= iter:
                        break
                    if best_validation_loss < obj_val_loss:
                        break
        self.loss_curve = np.append(self.loss_curve,np.array(curve))
        n_pts = int(self.loss_curve.shape[0]/2)
        self.loss_curve = self.loss_curve.reshape(n_pts,2)
        return self.w_best

    def CalROC(self, x_val, y_val):
        roc_dots = np.zeros((y_val.shape[0], 2))
        roc_dots[:,-1] = np.dot(x_val, self.w[0:-1]) + self.w[-1]
        roc_dots[:,0] = y_val
        roc_dots = roc_dots[np.lexsort(roc_dots.T)]
        (x, y) = (1, 1)
        roc_curve = [[x,y]]
        n_pos = sum(y_val)
        n_neg = x_val.shape[0] - n_pos
        for elem in roc_dots:
            if elem[0] == 1:
                y -= 1.0 / n_pos
            elif elem[0] == 0:
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

    def ReRoll(self):
        self.w *= 0
        self.epoch = 0
        self.loss_curve = np.array([])



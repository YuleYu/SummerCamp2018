import numpy as np
import copy
import LR.ext.Classifiers as Classifers


class Active_Fun:
    def __init__(self,fun,dfun,wmax):
        self.fun = fun
        self.dfun = dfun
        self.wmax = wmax

class Hidden_Layer:
    def __init__(self, n_prev_layer, n_this_layer, acf):
        w_max = acf.wmax(n_prev_layer,n_this_layer)
        self.w = np.random.random((n_prev_layer + 1, n_this_layer))
        self.w = 2 * w_max * self.w - w_max
        self.a = np.zeros(n_this_layer)
        self.z = np.zeros(n_this_layer)
        self.activate_fun = acf.fun
        self.d_activate_fun = acf.dfun
        self.delta = np.zeros((n_prev_layer + 1, n_this_layer))

    def forward_z_a(self, a):
        self.z = np.dot(a, self.w[0:-1, :]) + self.w[-1,:]
        self.a = self.activate_fun(self.z)
        return self.a

    def back_delta(self,next_W,next_delta):
        tt = np.dot(next_delta, next_W.T)
        self.delta = tt * self.d_activate_fun(self.a)

    def back_update(self, x, learning_rate, L2_reg):
        delta_W = -1.0 * np.dot(x.T, self.delta) / x.shape[0]
        delta_b = -1.0 * np.mean(self.delta, axis=0)
        self.w[0:-1] -= learning_rate * (L2_reg * self.w[0:-1] + delta_W)
        self.w[-1]   -= learning_rate * delta_b


class Output_Layer:
    def __init__(self, n_prev_layer, n_this_layer, acf):
        w_max = acf.wmax(n_prev_layer,n_this_layer)
        self.n_in = n_prev_layer
        self.n_out = n_this_layer
        self.w = np.random.random((n_prev_layer + 1, n_this_layer))
        self.w = 2 * w_max * self.w - w_max
        self.a = np.zeros(n_this_layer)
        self.z = np.zeros(n_this_layer)
        self.activate_fun = acf.fun
        self.d_activate_fun = acf.dfun
        self.delta = np.array([])

    def forward_y(self, a):
        self.z = np.dot(a, self.w[0:-1, :]) + self.w[-1,:]
        self.a = self.activate_fun(self.z)
        return self.a

    def back_compute_delta(self,y):
        yy = np.zeros((y.shape[0],self.n_out))
        yy[np.arange(yy.shape[0]),y] = 1.0
        self.delta = yy - self.a

    def back_update(self, x, learning_rate, L2_reg):
        delta_W = -1.0 * np.dot(x.T, self.delta) / x.shape[0]
        delta_b = -1.0 * np.mean(self.delta, axis=0)
        self.w[0:-1] -= learning_rate * (L2_reg * self.w[0:-1] + delta_W)
        self.w[-1]   -= learning_rate * delta_b



class MLP_Classifier(Classifers.Classifier):
    def __init__(self, layer_sizes, acfs):
        # layer_sizes: number of nodes in each layer
        # acfs : activation function of each layer
        self.hidden_layers = []
        self.n_layer = len(layer_sizes)
        for i in range(self.n_layer - 2):
            self.hidden_layers.append(Hidden_Layer(layer_sizes[i], layer_sizes[i + 1], acfs[i]))
        self.output_layer = Output_Layer(layer_sizes[-2],layer_sizes[-1],acfs[-1])
        self.epoch = 0
        self.L2_reg= 0.1

    def forward_feed(self, x):
        a = x
        for layer in self.hidden_layers:
            a = layer.forward_z_a(a)
        self.output_layer.forward_y(a)

    def back_propagation(self, x, y, learning_rate, L2_reg):
        self.output_layer.back_compute_delta(y)
        xx = self.hidden_layers[-1].a
        self.output_layer.back_update(xx, learning_rate, L2_reg)
        next_W = self.output_layer.w[0:-1]
        next_delta = self.output_layer.delta
        i = len(self.hidden_layers)
        while i > 0:
            curr_hidden_lay = self.hidden_layers[i-1]
            curr_hidden_lay.back_delta(next_W,next_delta)
            if i > 1:
                xx = self.hidden_layers[i-2].a
            else:
                xx = x
            curr_hidden_lay.back_update(xx,learning_rate,L2_reg)
            next_W = curr_hidden_lay.w[0:-1]
            next_delta = curr_hidden_lay.delta
            i -= 1

    def ZeroOneLoss(self, xs, ys):
        self.forward_feed(xs)
        a = self.output_layer.a
        n_err = np.sum([a[i,ys[i]] != a[i].max() for i in np.arange(ys.shape[0])])
        return n_err / ys.shape[0]

    def Loss(self, xs, ys):
        self.forward_feed(xs)
        yy = np.zeros((ys.shape[0],self.output_layer.n_out))
        yy[np.arange(yy.shape[0]),ys] = 1.0
        nll = -np.sum(yy*np.log(self.output_layer.a))
        return nll

    def Training(self, x_train, y_train, x_val, y_val, n_epoch, batchsize, learning_rate, patience,
                 validation_frequency, search_method, obj_val_loss=3.7e-3):
        # 参数初始化
        # 如果val_loss没有improvement_threshold以上的提升,不改变容忍度
        improvement_threshold = 1
        # 如果val_loss有improvement_threshold以上提升, 将容忍度提高为迭代数的1.2倍
        patience_increase = 1.5
        best_validation_loss = self.ZeroOneLoss(x_val, y_val)
        this_validation_loss = best_validation_loss
        self.best_hidden_layers = copy.deepcopy(self.hidden_layers)
        self.best_output_layer = copy.deepcopy(self.output_layer)
        done_looping = False
        curve = []
        epoch = 0

        # 主循环
        while epoch < n_epoch and (not done_looping):
            epoch += 1
            batch_pool = self.miniBatchInit(batchsize, x_train.shape[0])
            if search_method == 0: # Gradient without line search
                for batch_id in range(batch_pool.shape[0]):
                    minibatch_id = batch_pool[batch_id]
                    batch = x_train[minibatch_id]
                    batch_label = y_train[minibatch_id]
                    self.forward_feed(batch)
                    self.back_propagation(batch,batch_label,learning_rate,self.L2_reg)

                    iter = (epoch - 1) * batch_pool.shape[0] + batch_id
                    if (iter + 1) % validation_frequency == 0:
                        trainloss = self.Loss(x_train, y_train)
                        print( 'epoch: %i, minibatch: %i/%i, iter: %i, patience: %d, training loss: %f validation error %f %% best validation error %f %%' % (
                            epoch,
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
                            self.best_hidden_layers = copy.deepcopy(self.hidden_layers)
                            self.best_output_layer = copy.deepcopy(self.output_layer)
                        if patience <= iter:
                            done_looping = True
                            break
                        if best_validation_loss < obj_val_loss:
                            break

            # elif search_method == 2: #Newton method 牛顿法
            #     dloss = self.DLoss(x_train, y_train)
            #     ddloss = self.DDLoss(x_train)
            #     self.w -= solve(ddloss,dloss)
            #     iter = self.epoch - 1
            #     if (iter + 1) % validation_frequency == 0:
            #         trainloss = self.Loss(x_train, y_train)
            #         print('epoch: %i, iter: %i, patience: %d, training loss: %f validation error %f %%, best val_error: %f %%' % (
            #             self.epoch,
            #             iter,
            #             patience,
            #             trainloss,
            #             this_validation_loss * 100,
            #             best_validation_loss * 100))
            #         curve.append([iter, trainloss])
            #         this_validation_loss = self.ZeroOneLoss(x_val, y_val)
            #         if this_validation_loss < best_validation_loss:
            #             if this_validation_loss < best_validation_loss * improvement_threshold:
            #                 patience = max(patience, iter * patience_increase)
            #             best_validation_loss = this_validation_loss
            #             self.w_best = self.w
            #         if patience <= iter:
            #             break
            #         if best_validation_loss < obj_val_loss:
            #             break
        self.loss_curve = np.append(self.loss_curve,np.array(curve))
        self.epoch += epoch
        n_pts = int(self.loss_curve.shape[0]/2)
        self.loss_curve = self.loss_curve.reshape(n_pts,2)
        self.hidden_layers = self.best_hidden_layers
        self.output_layer = self.best_output_layer

    def calroc(self,x,y):
        self.forward_feed(x)
        roc_curve = []
        order = np.lexsort(self.output_layer.a.T)
        xp,yp = (1.0,1.0)
        n_pos = sum(y)
        n_neg = len(y) - n_pos
        for i in order:
            if y[i] == 1:
                yp -= 1.0 / n_pos
            else:
                xp -= 1.0 / n_neg
            roc_curve.append([xp,yp])
        return np.array(roc_curve)








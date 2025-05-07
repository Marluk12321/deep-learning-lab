import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


class TFDeep:
    def __init__(self, layer_sizes, param_delta=0.5, param_lambda=1e-3):
        """Arguments:
        - layer_sizes: sizes of each layer in the network
        - param_delta: training step
        - param_lambda: regularization
        """
        D = layer_sizes[0]
        C = layer_sizes[-1]

        # definicija podataka i parametara:
        self.X = tf.placeholder(tf.float32, [None, D])
        self.Yoh_ = tf.placeholder(tf.float32, [None, C])
        self.W = []
        self.b = []
        for D1, D2 in zip(layer_sizes, layer_sizes[1:]):
            i = len(self.W) + 1
            W = tf.Variable(0.1 * tf.random_normal([D1, D2]), name="W"+str(i))
            b = tf.Variable(0.1 * tf.random_normal([D2]), name="b"+str(i))
            self.W.append(W)
            self.b.append(b)

        # formulacija modela: izračunati self.probs
        self.h = []
        input = self.X
        for W, b in zip(self.W[:-1], self.b[:-1]):
            i = len(self.h) + 1
            h = tf.nn.relu(tf.add(tf.matmul(input, W), b))
            self.h.append(h)
            input = h
        W = self.W[-1]
        b = self.b[-1]
        input = self.h[-1] if len(self.h)>0 else self.X
        self.probs = tf.nn.softmax(tf.add(tf.matmul(input, W), b))

        # formulacija gubitka: self.loss
        self.reg_factor = param_lambda
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.Yoh_*tf.log(self.probs), reduction_indices=1))
        for W in self.W:
            self.loss += 0.5 * self.reg_factor * tf.reduce_sum(W**2)

        # formulacija operacije učenja: self.train_step
        self.train_step = param_delta
        self.trainer = tf.train.AdamOptimizer(self.train_step)
        self.train_op = self.trainer.minimize(self.loss)

        # instanciranje izvedbenog konteksta: self.session
        self.session = tf.Session()

    def train(self, X, Yoh_, param_niter):
        """Arguments:
        - X: actual datapoints [NxD]
        - Yoh_: one-hot encoded labels [NxC]
        - param_niter: number of iterations
        """
        # incijalizacija parametara
        self.session.run(tf.initialize_all_variables())

        vars = [self.loss, self.train_op]
        feed_dict = {self.X: X, self.Yoh_: Yoh_}

        # optimizacijska petlja
        for i in range(param_niter):

            loss_val, _ = self.session.run(vars, feed_dict)
            if (i+1) % 10 == 0:
                print("iteration {}: loss {}".format((i+1), loss_val))

    def train_es(self, X, Yoh_, param_niter, save_path):
        """Arguments:
        - X: actual datapoints [NxD]
        - Yoh_: one-hot encoded labels [NxC]
        - param_niter: number of iterations
        """
        # incijalizacija parametara
        self.session.run(tf.initialize_all_variables())

        saver = tf.train.Saver()
        X_train, X_test, y_train, y_test = train_test_split(X, Yoh_, test_size=0.2)

        train_vars = [self.loss, self.train_op]
        train_feed_dict = {self.X: X_train, self.Yoh_: y_train}
        test_feed_dict = {self.X: X_test, self.Yoh_: y_test}
        min_loss = None

        # optimizacijska petlja
        for i in range(param_niter):
            loss_val, _ = self.session.run(train_vars, train_feed_dict)
            if (i+1) % 10 == 0:
                print("iteration {}: loss {}".format((i+1), loss_val))
            
            loss_val = self.session.run(self.loss, test_feed_dict)
            if min_loss is None or loss_val < min_loss:
                min_loss = loss_val
                saver.save(self.session, save_path)

        saver.restore(self.session, save_path)

    def train_mb(self, X, Yoh_, param_niter, param_batches):
        """Arguments:
        - X: actual datapoints [NxD]
        - Yoh_: one-hot encoded labels [NxC]
        - param_niter: number of iterations
        """
        # incijalizacija parametara
        self.session.run(tf.initialize_all_variables())

        vars = [self.loss, self.train_op]
        kf = KFold(n_splits=param_batches, shuffle=True)

        # optimizacijska petlja
        for i in range(param_niter):
            for _, indices in kf.split(X):
                X_batch = X[indices]
                Y_batch = Yoh_[indices]
                feed_dict = {self.X: X_batch, self.Yoh_: Y_batch}
                loss_val, _ = self.session.run(vars, feed_dict)
            if (i+1) % 10 == 0:
                print("iteration {}: loss {}".format((i+1), loss_val))

    def eval(self, X):
        """Arguments:
        - X: actual datapoints [NxD]
        Returns: predicted class probabilites [NxC]
        """
        return self.session.run(self.probs, {self.X: X})


def tf_deep_decfun(model):
    return lambda X: np.argmax(model.eval(X), axis=1)

def count_params():
    for var in tf.trainable_variables():
        print(var.name)

if __name__ == "__main__":
    np.random.seed(100)
    tf.set_random_seed(100)
    D = 2
    K = 6
    C = 4
    N = 10
    param_niter = 10000
    param_delta = 0.1
    param_lambda = 1e-4
    layer_sizes = [D,10,10,C]

    # get the training dataset
    X, Y_ = data.sample_gmm_2d(K, C, N)

    # train the model
    model = TFDeep(layer_sizes, param_delta, param_lambda=1e-3)
    Yoh_ = OneHotEncoder().fit_transform(Y_.reshape((N*K,1))).toarray()
    model.train(X, Yoh_, param_niter)
    
    # evaluate the model on the training dataset
    probs = model.eval(X)
    Y = np.argmax(probs, axis=1)
    
    # report performance
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print ("accuracy recall precision:\n", accuracy, recall, precision)

    plt.figure()

    decfun = tf_deep_decfun(model)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=1, width=1000, height=1000)
    data.graph_data(X, Y_, Y)

    plt.show()

    count_params()

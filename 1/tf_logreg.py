import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data
from sklearn.preprocessing import OneHotEncoder


class TFLogreg:
  def __init__(self, D, C, param_delta=0.5, param_lambda=1e-3):
    """Arguments:
       - D: dimensions of each datapoint 
       - C: number of classes
       - param_delta: training step
       - param_lambda: regularization
    """

    # definicija podataka i parametara:
    self.X = tf.placeholder(tf.float64, [None, D])
    self.Yoh_ = tf.placeholder(tf.float64, [None, C])
    self.W = tf.Variable(2 * np.zeros((D, C)) - 1)
    self.b = tf.Variable(np.zeros((C,)))

    # formulacija modela: izračunati self.probs
    self.probs = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b)

    # formulacija gubitka: self.loss
    self.reg_factor = param_lambda
    self.loss = tf.reduce_mean(-tf.reduce_sum(self.Yoh_*tf.log(self.probs), reduction_indices=1)) \
                + 0.5 * self.reg_factor * tf.reduce_sum(self.W**2)

    # formulacija operacije učenja: self.train_step
    self.train_step = param_delta
    self.trainer = tf.train.GradientDescentOptimizer(self.train_step)
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

  def eval(self, X):
    """Arguments:
       - X: actual datapoints [NxD]
       Returns: predicted class probabilites [NxC]
    """
    return self.session.run(self.probs, {self.X: X})


def tf_logreg_decfun(model):
    return lambda X: np.argmax(model.eval(X), axis=1)

if __name__ == "__main__":
    np.random.seed(100)
    tf.set_random_seed(100)
    D = 2
    K = 6
    C = 3
    N = 10
    param_niter = 10000
    param_delta = 0.1
    param_lambda = 1e-3
    hidden_layer_size = 5

    # get the training dataset
    X, Y_ = data.sample_gmm_2d(K, C, N)

    # train the model
    model = TFLogreg(D, C, param_delta, param_lambda=1e-3)
    Yoh_ = OneHotEncoder().fit_transform(Y_.reshape((N*K,1))).toarray()
    model.train(X, Yoh_, param_niter)
    
    # evaluate the model on the training dataset
    probs = model.eval(X)
    Y = np.argmax(probs, axis=1)
    
    # report performance
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print ("accuracy recall precision:\n", accuracy, recall, precision)

    plt.figure()

    decfun = tf_logreg_decfun(model)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=1, width=1000, height=1000)
    data.graph_data(X, Y_, Y)

    plt.show()

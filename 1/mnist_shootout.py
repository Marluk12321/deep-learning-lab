import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tf_deep import *
from ksvm_wrap import *
import numpy as np

tf.app.flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
mnist = input_data.read_data_sets(tf.app.flags.FLAGS.data_dir, one_hot=True)
print()

N=mnist.train.images.shape[0]
D=mnist.train.images.shape[1]
C=mnist.train.labels.shape[1]

# get the training dataset
X  = mnist.train.images
Y_ = mnist.train.labels
Y__ = np.argmax(Y_, axis=1)

# get the testing dataset
X_test = mnist.test.images
Y_test_ = mnist.test.labels
Y_test__ = np.argmax(Y_test_, axis=1)

print("TFDeep")

param_niter = 1000
#param_delta = 1e-4
param_delta = tf.train.exponential_decay( \
    learning_rate=1e-4, decay_rate=1-1e-4, global_step=1, decay_steps=1)
param_lambda = 1e-4
layer_sizes = [D,100,C]

# train the model
model = TFDeep(layer_sizes, param_delta, param_lambda=1e-3)
#model.train(X, Y_, param_niter)
#model.train_es(X, Y_, param_niter, "model.ckpt")
model.train_mb(X, Y_, 100, 100)

print("Train data performance:")

# evaluate the model on the training dataset
probs = model.eval(X)
Y = np.argmax(probs, axis=1)

# report performance
accuracy, recall, precision = data.eval_perf_multi(Y, Y__)
print ("accuracy recall precision:\n", accuracy, recall, precision)
print()

print("Test data performance:")

# evaluate the model on the training dataset
probs = model.eval(X_test)
Y_test = np.argmax(probs, axis=1)

# report performance
accuracy, recall, precision = data.eval_perf_multi(Y_test, Y_test__)
print ("accuracy recall precision:\n", accuracy, recall, precision)
print()

print("KSVM")

param_svmc = 100
param_svm_gamma = "auto"

# train the model
print("training SVM...")
model = KSVMWrap(X, Y__, param_svmc, param_svm_gamma)

# evaluate the model on the training dataset
Y = model.predict(X)

# report performance
accuracy, recall, precision = data.eval_perf_multi(Y, Y__)
print ("accuracy recall precision:\n", accuracy, recall, precision)
print()

print("Test data performance:")

# evaluate the model on the training dataset
Y_test = model.predict(X_test)

# report performance
accuracy, recall, precision = data.eval_perf_multi(Y_test, Y_test__)
print ("accuracy recall precision:\n", accuracy, recall, precision)
print()

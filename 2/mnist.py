import tensorflow as tf
import numpy as np
from tfnn import *


DATA_DIR = "/mnt/hgfs/Google Drive/sk/du/lab/2/data"
SAVE_DIR = "/mnt/hgfs/Google Drive/sk/du/lab/2/save/mnist"


config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['weight_decay'] = 1e-3
config['lr_policy'] = {0:1e-1, 2:1e-2, 4:1e-3, 6:1e-4}
config['conv1_scope'] = "conv1"


def build_model(inputs, labels, num_classes, weight_decay, conv1sz, conv2sz, fc3sz):
    net = convolution_relu(inputs, conv1sz, weight_decay=weight_decay, scope="conv1")
    net = maxpool(net, scope="pool1")
    net = convolution_relu(net, conv2sz, weight_decay=weight_decay, scope="conv2")
    net = maxpool(net, scope="pool2")
    net = layers.flatten(net)
    net = fully_connected_relu(net, fc3sz, weight_decay, scope="fc3")

    logits = fully_connected_logits(net, num_classes, scope="logits")
    loss = mean_softmax_loss(logits, labels)

    return logits, loss

#np.random.seed(100)
np.random.seed(int(time.time() * 1e6) % 2**31)
dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)
train_x = dataset.train.images
train_x = train_x.reshape([-1, 1, 28, 28]).transpose(0,2,3,1)
train_y = dataset.train.labels
valid_x = dataset.validation.images
valid_x = valid_x.reshape([-1, 1, 28, 28]).transpose(0,2,3,1)
valid_y = dataset.validation.labels
test_x = dataset.test.images
test_x = test_x.reshape([-1, 1, 28, 28]).transpose(0,2,3,1)
test_y = dataset.test.labels
train_mean = train_x.mean()
train_x -= train_mean
valid_x -= train_mean
test_x -= train_mean

inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])
num_classes = 10
labels = tf.placeholder(tf.float32, [None, num_classes])
weight_decay = config['weight_decay']
conv1sz = 16
conv2sz = 32
fc3sz = 512

logits, loss = build_model(\
    inputs, labels, num_classes, weight_decay, conv1sz, conv2sz, fc3sz)

global_step = tf.Variable(0, trainable=False)
'''
learning_rate = tf.train.exponential_decay( \
    learning_rate=1e-4, decay_rate=1-1e-4, \
    global_step=global_step, decay_steps=1)
'''
#'''
learning_rate = tf.train.exponential_decay( \
    learning_rate=1e-1, decay_rate=1e-1, \
    global_step=global_step, decay_steps=2, staircase=True)
#'''
#learning_rate = 1e-4
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

train(logits, loss, optimizer, global_step, learning_rate, inputs, labels, \
    train_x, train_y, valid_x, valid_y, test_x, test_y, config)

import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tfnn import *
from sklearn.preprocessing import OneHotEncoder
import skimage as ski
import skimage.io
import heapq


def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def draw_image(img, mean, std):
    img *= std
    img += mean
    img = img.astype(np.uint8)
    ski.io.imshow(img)
    ski.io.show()


DATA_DIR = "/mnt/hgfs/Google Drive/sk/du/lab/2/data/cifar-10-batches-py"
SAVE_DIR = "/mnt/hgfs/Google Drive/sk/du/lab/2/save/cifar"

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['weight_decay'] = 1e-3
config['lr_policy'] = {0:1e-1, 2:1e-2, 4:1e-3, 6:1e-4}
config['conv1_scope'] = "conv1"
config['learning_rate'] = 1e-2 / config['batch_size']
config['decay_rate'] = 1 - 1e-2


# ***** data loading *****
img_height = 32
img_width = 32
num_channels = 3
num_classes = 10

print("loading data...")
train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
    subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
    train_x = np.vstack((train_x, subset['data']))
    train_y += subset['labels']
train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1)
train_y = np.array(train_y, dtype=np.int32)
train_y = OneHotEncoder().fit_transform(train_y.reshape((-1,1))).toarray()

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)
test_y = OneHotEncoder().fit_transform(test_y.reshape((-1,1))).toarray()

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]
train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]
data_mean = train_x.mean((0,1,2))
data_std = train_x.std((0,1,2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std

# ***** net config *****
inputs = tf.placeholder(tf.float32, [None, img_height, img_width, num_channels])
labels = tf.placeholder(tf.float32, [None, num_classes])
weight_decay = config['weight_decay']

net = convolution_relu(inputs, 16, 5, weight_decay, scope=config["conv1_scope"])
net = maxpool(net, 3, 2, scope="pool1")
net = convolution_relu(net, 32, 5, weight_decay, scope="conv2")
net = maxpool(net, 3, 2, scope="pool2")
net = layers.flatten(net)
net = fully_connected_relu(net, 128, weight_decay, scope="fc1")
net = fully_connected_logits(net, num_classes, scope="logits")

loss = mean_softmax_loss(net, labels)

# ***** training *****
global_step = tf.Variable(0, trainable=False)
#'''
learning_rate = tf.train.exponential_decay( \
    learning_rate=config["learning_rate"], \
    decay_rate=config["decay_rate"], \
    global_step=global_step, decay_steps=1)
#'''
'''
learning_rate = tf.train.exponential_decay( \
    learning_rate=1e-1, decay_rate=1e-1, \
    global_step=global_step, decay_steps=2, staircase=True)
'''
#learning_rate = 1e-4
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

print("training...")
session = train(net, loss, optimizer, global_step, learning_rate, inputs, labels, \
    train_x, train_y, valid_x, valid_y, test_x, test_y, \
    config)


# ***** drawing worst *****
print("drawing 20 worst...")

heap = []
for i in range(test_x.shape[0]):
    test_xi = test_x[i].reshape((1, img_height, img_width, num_channels))
    test_yi = test_y[i].reshape((1, num_classes))
    feed_dict = {inputs: test_xi, labels: test_yi}
    preds, loss_val = session.run([net, loss], feed_dict)

    pred_class = np.argmax(preds[0])
    correct_class = np.argmax(test_y[i])
    if pred_class != correct_class:
        heapq.heappush(heap, (loss_val, i, preds[0]))
        if len(heap) > 20:
            heapq.heappop(heap)

for _, i, preds in heap:
    draw_image(test_x[i], data_mean, data_std)
    correct_class = np.argmax(test_y[i])
    third_best = min(heapq.nlargest(3, preds))
    top_classes = [i for i, pred in enumerate(preds) if pred >= third_best]
    print("Correct class: {}, top 3 classes: {}".format(correct_class, top_classes))

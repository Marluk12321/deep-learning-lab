import numpy as np
import time
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import KFold
import skimage as ski
import skimage.io
import os, os.path
import matplotlib.pyplot as plt
import warnings


def convolution_relu(inputs, num_outputs, kernel_size=5, weight_decay=0.0, scope=None):
    stride = 1
    padding = 'SAME'
    activation_fn = tf.nn.relu
    weights_initializer = layers.variance_scaling_initializer()
    weights_regularizer = layers.l2_regularizer(weight_decay)
    return layers.convolution2d(inputs=inputs, num_outputs=num_outputs, \
        kernel_size=kernel_size, stride=stride, padding=padding, \
        activation_fn=activation_fn, weights_initializer=weights_initializer, \
        weights_regularizer=weights_regularizer, scope=scope)

def maxpool(inputs, kernel_size=2, stride=2, scope=None):
    return layers.max_pool2d(inputs=inputs, kernel_size=kernel_size, \
        stride=stride, scope=scope)

def fully_connected_relu(inputs, num_outputs, weight_decay=0.0, scope=None):
    activation_fn = tf.nn.relu
    weights_initializer = layers.variance_scaling_initializer()
    weights_regularizer = layers.l2_regularizer(weight_decay)
    return layers.fully_connected(inputs=inputs, num_outputs=num_outputs, \
        activation_fn=activation_fn, weights_initializer=weights_initializer, \
        weights_regularizer=weights_regularizer, scope=scope)

def fully_connected_logits(inputs, num_outputs, scope=None):
    return layers.fully_connected(inputs=inputs, num_outputs=num_outputs, \
        activation_fn=None, scope=scope)

def mean_softmax_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( \
        logits=logits, labels=labels))


def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[3]
    num_channels = w.shape[2]
    if num_channels not in [1, 3, 4]:
        return
    k = w.shape[0]
    assert w.shape[0] == w.shape[1]
    w = w.reshape(k, k, num_channels, num_filters)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = int(np.ceil(num_filters / cols))
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k, c:c+k, :] = w[:,:,:,i]
    if num_channels == 1:
        img = img.reshape((height, width))
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ski.io.imsave(os.path.join(save_dir, filename), img)

def plot_training_progress(save_dir, data):
    _, ((ax1, ax2), (ax3, _)) = plt.subplots(2, 2, figsize=(16,8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color, \
            linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color, \
            linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color, \
            linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color, \
            linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color, \
            linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot.pdf')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)

def train(logits, loss, optimizer, global_step, learning_rate, inputs, labels, \
          train_x, train_y, valid_x, valid_y, test_x, test_y, config):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    optimization_op = optimizer.minimize(loss)
    session = tf.Session()
    session.run(tf.initialize_all_variables())

    param_niter = config["max_epochs"]
    batch_size = config['batch_size']
    n_batches = train_x.shape[0] // batch_size
    kf = KFold(n_splits=n_batches, shuffle=True)

    plot_data = {}
    plot_data['train_loss'] = []
    plot_data['valid_loss'] = []
    plot_data['train_acc'] = []
    plot_data['valid_acc'] = []
    plot_data['lr'] = []

    for i in range(param_niter):
        session.run(global_step.assign(i))

        # batch training
        for _, indices in kf.split(train_x):
            batch_x = train_x[indices]
            batch_y = train_y[indices]
            feed_dict = {inputs: batch_x, labels: batch_y}
            loss_val, _ = session.run([loss, optimization_op], feed_dict)

        # batch evaluation
        total_loss = 0.0
        total_acc = 0.0
        for _, indices in kf.split(train_x):
            batch_x = train_x[indices]
            batch_y = train_y[indices]
            feed_dict = {inputs: batch_x, labels: batch_y}
            loss_val, acc_val = session.run([loss, accuracy], feed_dict)
            total_loss += loss_val
            total_acc += acc_val

        # epoch report
        print("iteration {}:".format(i+1))

        train_loss = total_loss / n_batches
        train_acc = total_acc / n_batches
        print("\ttrain loss {}, train accuracy: {}".format(train_loss, train_acc))

        feed_dict = {inputs: valid_x, labels: valid_y}
        valid_loss, valid_acc = session.run([loss, accuracy], feed_dict)
        print("\tvalidation loss {}, validation accuracy: {}".format(valid_loss, valid_acc))

        # filter2img
        if "conv1_scope" in config:
            conv1_var = tf.contrib.framework.get_variables(config["conv1_scope"])[0]
            conv1_weights = conv1_var.eval(session=session)
            draw_conv_filters(i+1, 0, conv1_weights, config['save_dir'])

        # plot data
        plot_data['train_loss'].append(train_loss)
        plot_data['valid_loss'].append(valid_loss)
        plot_data['train_acc'].append(train_acc)
        plot_data['valid_acc'].append(valid_acc)
        plot_data['lr'].append(learning_rate.eval(session=session))
    plot_training_progress(config['save_dir'], plot_data)
    plt.clf()
    print()

    # final report
    feed_dict = {inputs: test_x, labels: test_y}
    test_loss, test_acc = session.run([loss, accuracy], feed_dict)
    print("test loss {}, test accuracy: {}\n".format(test_loss, test_acc))

    return session

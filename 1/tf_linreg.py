import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def calc_grad_a(X, Y, Y_):
    N = X.shape[0]
    return 2 * np.sum((Y-Y_) * X)

def calc_grad_b(X, Y, Y_):
    N = X.shape[0]
    return 2 * np.sum(Y-Y_)

## 1. definicija računskog grafa
# podatci i parametri
X  = tf.placeholder(tf.float32, [None])
Y_ = tf.placeholder(tf.float32, [None])
a = tf.Variable(0.0)
b = tf.Variable(0.0)

# afini regresijski model
Y = a * X + b

# kvadratni gubitak
loss = (Y-Y_)**2

# optimizacijski postupak: gradijentni spust
trainer = tf.train.GradientDescentOptimizer(0.01)
#train_op = trainer.minimize(loss)
grads_and_vars = trainer.compute_gradients(loss, [a, b])
train_op = trainer.apply_gradients(grads_and_vars)

grad_a = grads_and_vars[0][0]
grad_b = grads_and_vars[1][0]

grad_a_ = tf.Print(grad_a, [grad_a], "grad_a: ")
grad_b_ = tf.Print(grad_b, [grad_b], "grad_b: ")

xs = np.array([1,2,3])
ys_ = np.array([3,5,7])

vars = [loss, train_op, a,b, grad_a_,grad_b_]
feed_dict = {X: xs, Y_: ys_}

## 2. inicijalizacija parametara
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables())

## 3. učenje
# neka igre počnu!
for i in range(500):
    val_loss, _, val_a,val_b, val_grad_a, val_grad_b = sess.run(vars, feed_dict)
    print((i+1),val_loss, val_a,val_b)
    ys = np.array([val_a*x + val_b for x in xs])
    print("\t", val_grad_a,val_grad_b, calc_grad_a(xs, ys, ys_), calc_grad_b(xs, ys, ys_))

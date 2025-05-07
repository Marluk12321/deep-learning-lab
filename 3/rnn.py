import os
import numpy as np
from dataset import Dataset



class RNN:
    def __init__(self, vocab_size, hidden_size=100, sequence_length=30, learning_rate=1e-1):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        l = 0.0     # loc
        s = 1e-2    # scale

        self.U = np.random.normal(l, s, (vocab_size, hidden_size))  # input projection
        self.W = np.random.normal(l, s, (hidden_size, hidden_size)) # hidden-to-hidden projection
        self.b = np.zeros((hidden_size,))                           # input bias

        self.V = np.random.normal(l, s, (hidden_size, vocab_size))  # output projection
        self.c = np.zeros((vocab_size,))                            # output bias

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U = np.zeros(self.U.shape)
        self.memory_W = np.zeros(self.W.shape)
        self.memory_b = np.zeros(self.b.shape)

        self.memory_V = np.zeros(self.V.shape)
        self.memory_c = np.zeros(self.c.shape)

    @staticmethod
    def rnn_step_forward(x, h_prev, U, W, b):
        '''
        # A single time step forward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # x - input data (minibatch size x input dimension)
        # h_prev - previous hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)
        '''

        h_current = np.tanh(x.dot(U) + h_prev.dot(W) + b)
        cache = (x, h_prev, W, h_current)

        # return the new hidden state and a tuple of values needed for the backward step
        return h_current, cache

    @staticmethod
    def rnn_forward(x, h0, U, W, b):
        '''
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        # x - input data for the whole time-series
        #     (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)
        '''

        cache = []
        h = h0
        for i in range(0, x.shape[1]):  # for each sequence
            xi = x[:, i, :].reshape(x.shape[0], x.shape[2])
            h, ci = RNN.rnn_step_forward(xi, h, U, W, b)
            cache.append(ci)

        # return the hidden states for the whole time series (T+1)
        # and a tuple of values needed for the backward step
        return h, cache

    @staticmethod
    def rnn_step_backward(grad_next, cache):
        '''
        # A single time step backward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # grad_next - upstream gradient of the loss with respect to
        #             the next hidden state and current output
        # cache - cached information from the forward pass
        '''

        x, h_prev, W, h_current = cache
        dL_da = grad_next * (1 - h_current**2)

        dh_prev = dL_da.dot(W)
        dU = x.T.dot(dL_da)
        dW = h_prev.T.dot(dL_da)
        db = np.sum(dL_da, axis=0)

        return dh_prev, dU, dW, db


    def rnn_backward(self, dh, cache):
        '''
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearit
        '''

        dU = np.zeros(self.U.shape)
        dW = np.zeros(self.W.shape)
        db = np.zeros(self.b.shape)

        for ci in reversed(cache):
            dh_prev, dUi, dWi, dbi = RNN.rnn_step_backward(dh, ci)
            dh = dh_prev
            dU += dUi
            dW += dWi
            db += dbi

        # compute and return gradients with respect to each parameter
        # for the whole time series.
        return dU, dW, db

    @staticmethod
    def output(h, V, c):
        # Calculate the output probabilities of the network
        return softmax(h.dot(V) + c)

    @staticmethod
    def output_loss_and_grads(h, V, c, y):
        '''
        # Calculate the loss of the network for each of the outputs

        # h - hidden states of the network for each timestep.
        #     the dimensionality of h is (batch size x sequence length x hidden size
        #     (the initial state is irrelevant for the output)
        # V - the output projection matrix of dimension hidden size x vocabulary size
        # c - the output bias of dimension vocabulary size x 1
        # y - the true class distribution - a one-hot vector of dimension
        #     vocabulary size x 1
        '''

        outputs = RNN.output(h, V, c)

        loss = -np.sum(y * np.log(outputs))
        dL_do = outputs - y

        dh = dL_do.dot(V.T)
        dV = h.T.dot(dL_do)
        dc = np.sum(dL_do, axis=0)

        return loss, dh, dV, dc

    def calc_update_value(self, memory, grad):
        a_min = -5
        a_max = 5
        delta = 1e-7

        value = -(self.learning_rate / (delta + np.sqrt(memory))) * grad
        value = np.clip(value, a_min, a_max)

        return value

    def update(self, dU, dW, db, dV, dc):

        # update memory matrices
        self.memory_U += dU**2
        self.memory_W += dW**2
        self.memory_b += db**2
        self.memory_V += dV**2
        self.memory_c += dc**2

        # perform the Adagrad update of parameters
        self.U += self.calc_update_value(self.memory_U, dU)
        self.W += self.calc_update_value(self.memory_W, dW)
        self.b += self.calc_update_value(self.memory_b, db)
        self.V += self.calc_update_value(self.memory_V, dV)
        self.c += self.calc_update_value(self.memory_c, dc)

    def step(self, h0, x_oh, y_oh):
        h, cache = self.rnn_forward(x_oh, h0, self.U, self.W, self.b)
        loss, dh, dV, dc = self.output_loss_and_grads(h, self.V, self.c, y_oh[:, -1])
        dU, dW, db = self.rnn_backward(dh, cache)
        self.update(dU, dW, db, dV, dc)
        return loss, h

    def sample(self, seed, n_sample, dataset):
        # inicijalizirati h0 na vektor nula
        h0 = np.zeros((1, self.hidden_size))
        # seed string pretvoriti u one-hot reprezentaciju ulaza
        sample = dataset.encode(seed)
        seed_onehot = dataset.one_hot_encode(sample)
        seed_onehot = seed_onehot.reshape((1,) + seed_onehot.shape)

        h = h0
        h, _ = self.rnn_forward(seed_onehot, h0, self.U, self.W, self.b)

        while len(sample) < n_sample:
            outputs = self.output(h, self.V, self.c)
            sample = np.append(sample, np.argmax(outputs[0]))
            next_onehot = dataset.one_hot_encode(sample[-1:])
            h, _ = self.rnn_step_forward(next_onehot, h, self.U, self.W, self.b)

        return "".join(dataset.decode(sample))



def softmax(x):
    max_x = np.max(x)
    e_x = np.exp(x - max_x)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def main():
    root = "data"
    dataset_name = "selected_conversations.txt"
    dataset_path = os.path.join(root, dataset_name)
    dataset = Dataset(20, 30, dataset_path)
    minibatch = dataset.next_minibatch()

    rnn = RNN(len(dataset.sorted_chars))
    h0 = np.zeros((dataset.batch_size, rnn.hidden_size))
    x = minibatch[1]
    xoh = []
    for xi in x:
        xoh.append(dataset.one_hot_encode(xi))
    xoh = np.array(xoh)
    h, cache = rnn.rnn_forward(xoh, h0, rnn.U, rnn.W, rnn.b)
    print(h.shape)

    y = minibatch[2]
    yoh = dataset.one_hot_encode(y[:, -1])
    loss, dh, dV, dc = rnn.output_loss_and_grads(h, rnn.V, rnn.c, yoh)
    print(loss, dh.shape, h.shape, dV.shape, rnn.V.shape, dc.shape, rnn.c.shape)

    dU, dW, db = rnn.rnn_backward(dh, cache)
    print(dU.shape, dW.shape, db.shape)

    rnn.update(dU, dW, db, dV, dc)

if __name__ == "__main__":
    main()

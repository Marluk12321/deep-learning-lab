import numpy as np
import data
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


def relu(X):
    return np.maximum(0, X)

def softmax(X):
    expX = np.exp(X)
    sumexp = np.sum(expX, axis=1, keepdims=True)
    probs = expX / sumexp
    return probs

def fcann2_train(X, Y_, hidden_layer_size, param_niter=500, param_delta=0.1, param_lambda=0.002):
    '''
    Argumenti
        X:  podatci, np.array NxD
        Y_: indeksi razreda, np.array NxC
        param_niter, param_delta: parametri gradijentnog spusta

    Povratne vrijednosti
        w, b: parametri neuronske mreže
    '''

    N = X.shape[0]
    D = X.shape[1]
    C = np.max(Y_) + 1
    H = hidden_layer_size

    Y_encoded = OneHotEncoder().fit_transform(Y_.reshape((N,1))).toarray()

    W1 = 0.01 * np.random.randn(D, H)    # DxH
    b1 = np.zeros((H,))                 # Hx1
    W2 = 0.01 * np.random.randn(H, C)    # HxC
    b2 = np.zeros((C,))                 # Cx1

    # gradijentni spust (param_niter iteracija)
    for i in range(param_niter):
        s1 = X.dot(W1) + b1     # NxH
        h1 = relu(s1)           # NxH
        s2 = h1.dot(W2) + b2    # NxC
        probs = softmax(s2)     # NxC

        logprobs = np.log(np.abs(1-Y_encoded-probs))    # N x C
        loss = np.sum(-logprobs) / N    # scalar
        loss += 0.5 * param_lambda * (np.sum(W1*W1) + np.sum(W2*W2))

        # dijagnostički ispis
        if (i+1) % 10 == 0:
            print("iteration {}: loss {}".format((i+1), loss))

        dL_ds = (probs - Y_encoded) / N   # NxC

        grad_W2 = h1.T.dot(dL_ds)         # CxD
        grad_W2 += param_lambda * W2
        grad_b2 = np.sum(dL_ds, axis=0)   # Cx1

        dL_dh1 = dL_ds.dot(W2.T)          # NxH
        dL_dh1[h1 <= 0] = 0

        grad_W1 = X.T.dot(dL_dh1)         # DxH
        grad_W1 += param_lambda * W1
        grad_b1 = np.sum(dL_dh1, axis=0)  # Hx1

        W2 += -param_delta * grad_W2
        b2 += -param_delta * grad_b2

        W1 += -param_delta * grad_W1
        b1 += -param_delta * grad_b1

    return W1, b1, W2, b2

def fcann2_classify(X, W1, b1, W2, b2):
    '''
    Argumenti
        X:              podatci, np.array NxD
        W1, b1, W2, b2: parametri neuronske mreže

    Povratne vrijednosti
        probs: vjerojatnosti razreda, NxC
    '''
    s1 = X.dot(W1) + b1     # NxH
    h1 = relu(s1)           # NxH
    s2 = h1.dot(W2) + b2    # NxC
    probs = softmax(s2)     # NxC

    return probs

def fcann2_decfun(W1, b1, W2, b2):
    def classify(X):
        scores = fcann2_classify(X, W1, b1, W2, b2)
        return np.argmax(scores, axis=1)
    return classify


if __name__ == "__main__":
    np.random.seed(100)
    K = 6
    C = 2
    N = 10
    param_niter = int(1e5)
    param_delta = 0.1
    param_lambda = 1e-3
    hidden_layer_size = 5

    # get the training dataset
    X, Y_ = data.sample_gmm_2d(K, C, N)

    # train the model
    W1, b1, W2, b2 = fcann2_train(X, Y_, hidden_layer_size, \
        param_niter=param_niter, param_delta=param_delta, param_lambda=param_lambda)
    
    # evaluate the model on the training dataset
    probs = fcann2_classify(X, W1, b1, W2, b2)
    Y = np.argmax(probs, axis=1)
    
    # report performance
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print ("accuracy recall precision:\n", accuracy, recall, precision)

    plt.figure()

    decfun = fcann2_decfun(W1, b1, W2, b2)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=1, width=1000, height=1000)
    data.graph_data(X, Y_, Y)

    plt.show()

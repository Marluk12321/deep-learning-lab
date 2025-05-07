import numpy as np
import data
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


def logreg_train(X, Y_, param_niter=500, param_delta=0.1):
    '''
    Argumenti
        X:  podatci, np.array NxD
        Y_: indeksi razreda, np.array NxC
        param_niter, param_delta: parametri gradijentnog spusta

    Povratne vrijednosti
        w, b: parametri logističke regresije
    '''

    N = X.shape[0]
    D = X.shape[1]
    C = np.max(Y_) + 1

    Y_encoded = OneHotEncoder().fit_transform(Y_.reshape((N,1))).toarray()

    W = 2 * np.random.rand(D, C) - 1.0
    b = np.zeros((C,))

    # gradijentni spust (param_niter iteracija)
    for i in range(param_niter):
        # eksponencirani klasifikacijski rezultati
        scores = X.dot(W) + b               # N x C
        expscores = np.exp(scores)          # N x C

        # nazivnik sofmaksa
        sumexp = np.sum(expscores, axis=1, keepdims=True)  # N x 1

        # logaritmirane vjerojatnosti razreda 
        probs = expscores / sumexp  # N x C
        logprobs = np.log(np.abs(1-Y_encoded-probs))    # N x C

        # gubitak
        loss = np.sum(-logprobs)    # scalar

        # dijagnostički ispis
        if (i+1) % 10 == 0:
            print("iteration {}: loss {}".format((i+1), loss))

        # derivacije komponenata gubitka po rezultatu
        dL_ds = probs - Y_encoded   # N x C

        # gradijenti parametara
        grad_W = dL_ds.T.dot(X).T       # C x D (ili D x C)
        grad_b = np.sum(dL_ds, axis=0)  # C x 1 (ili 1 x C)

        # poboljšani parametri
        W += -param_delta * grad_W
        b += -param_delta * grad_b

    return W, b

def logreg_classify(X, W, b):
    '''
    Argumenti
        X:    podatci, np.array NxD
        W, b: parametri logističke regresije

    Povratne vrijednosti
        probs: vjerojatnosti razreda, NxC
    '''
    # eksponencirani klasifikacijski rezultati
    scores = X.dot(W) + b               # N x C
    expscores = np.exp(scores)          # N x C

    # nazivnik sofmaksa
    sumexp = np.sum(expscores, axis=1, keepdims=True)  # N x 1

    # logaritmirane vjerojatnosti razreda 
    probs = expscores / sumexp  # N x C

    return probs

def logreg_decfun(W, b):
    def classify(X):
        scores = logreg_classify(X, W, b)
        return np.argmax(scores, axis=1)
    return classify


if __name__ == "__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(3, 100)

    # train the model
    W, b = logreg_train(X, Y_)
    
    # evaluate the model on the training dataset
    probs = logreg_classify(X, W, b)
    Y = np.argmax(probs, axis=1)
    
    # report performance
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print ("accuracy recall precision:\n", accuracy, recall, precision)

    plt.figure()

    decfun = logreg_decfun(W, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=1, width=1000, height=1000)
    data.graph_data(X, Y_, Y)

    plt.show()

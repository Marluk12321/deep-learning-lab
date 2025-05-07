import numpy as np
import data


def sigma(x):
    ex = np.e ** x
    return ex / (1 + ex)
sigma = np.vectorize(sigma, otypes=[np.float])

def single_logloss(p, y_):
    sigma_p = p if y_ == 1 else 1-p
    return -np.log(sigma_p)
single_logloss = np.vectorize(single_logloss, otypes=[np.float])

def logloss(probs, Y_):
    return np.sum(single_logloss(probs, Y_))

def binlogreg_train(X, Y_, param_niter=500, param_delta=0.1):
    '''
    Argumenti
        X:  podatci, np.array Nx2
        Y_: indeksi razreda, np.array Nx1
        param_niter, param_delta: parametri gradijentnog spusta

    Povratne vrijednosti
        w, b: parametri logističke regresije
    '''

    N = X.shape[0]
    D = X.shape[1]

    w = 2 * np.random.random_sample(D) - 1.0
    b = 0.0

    # gradijentni spust (param_niter iteracija)
    for i in range(param_niter):
        # klasifikacijski rezultati
        scores = X.dot(w) + b # N x 1
        
        # vjerojatnosti razreda c_1
        probs = sigma(scores) # N x 1

        # gubitak
        loss = logloss(probs, Y_) # scalar
        
        # dijagnostički ispis
        if (i+1) % 10 == 0:
            print("iteration {}: loss {}".format((i+1), loss))

        # derivacije gubitka po klasifikacijskom rezultatu
        dL_dscores = probs - Y_      # N x 1

        # gradijenti parametara
        grad_w = dL_dscores.T.dot(X) # D x 1
        grad_b = np.sum(dL_dscores)  # 1 x 1

        # poboljšani parametri
        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b

def binlogreg_classify(X, w, b):
    '''
    Argumenti
        X:    podatci, np.array Nx2
        w, b: parametri logističke regresije

    Povratne vrijednosti
        probs: vjerojatnosti razreda c1
    '''

    # klasifikacijski rezultati
    scores = X.dot(w) + b # N x 1
    
    # vjerojatnosti razreda c_1
    probs = sigma(scores) # N x 1

    return probs


if __name__ == "__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 100)

    # train the model
    w, b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = np.around(probs)

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print ("accuracy recall precision AP:\n", accuracy, recall, precision, AP)

    data.graph_data(X, Y_, Y)



def binlogreg_decfun(w, b):
    def classify(X):
        return binlogreg_classify(X, w, b)
    return classify

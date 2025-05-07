import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import data
import tf_deep
from tensorflow import set_random_seed
from sklearn.preprocessing import OneHotEncoder


class KSVMWrap:
    def __init__(self, X, Y_, c=1, g='auto'):
        '''
        Konstruira omotač i uči RBF SVM klasifikator
        X,Y_: podatci i točni indeksi razreda
        c:    relativni značaj podatkovne cijene
        g:    širina RBF jezgre
        '''
        self.model = SVC(C=c, gamma=g)
        self.model.fit(X, Y_)

    def predict(self, X):
        '''
        Predviđa i vraća indekse razreda podataka X
        '''
        return self.model.predict(X)

    def get_scores(self, X):
        '''
        Vraća klasifikacijske rezultate podataka X
        '''
        return self.model.decision_function(X)

    def support(self):
        '''
        Indeksi podataka koji su odabrani za potporne vektore
        '''
        return self.model.support_

    
def ksvm_decfun(model):
    return lambda X: model.predict(X)

if __name__ == "__main__":
    np.random.seed(100)
    D = 2
    K = 6
    C = 3
    N = 10

    print("SVM")

    param_svmc = 100
    param_svm_gamma = "auto"

    # get the training dataset
    X, Y_ = data.sample_gmm_2d(K, C, N)

    # train the model
    model = KSVMWrap(X, Y_, param_svmc, param_svm_gamma)
    #Yoh_ = OneHotEncoder().fit_transform(Y_.reshape((N*K,1))).toarray()
    
    # evaluate the model on the training dataset
    Y = model.predict(X)
    
    # report performance
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print ("accuracy recall precision:\n", accuracy, recall, precision)

    plt.figure()

    decfun = ksvm_decfun(model)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=1, width=1000, height=1000)
    data.graph_data(X, Y_, Y, special=model.support())

    plt.show()


    print("TFDeep")

    set_random_seed(100)
    param_niter = 10000
    param_delta = 0.1
    param_lambda = 1e-4
    layer_sizes = [D,10,10,C]

    # train the model
    model = tf_deep.TFDeep(layer_sizes, param_delta, param_lambda=1e-3)
    Yoh_ = OneHotEncoder().fit_transform(Y_.reshape((N*K,1))).toarray()
    model.train(X, Yoh_, param_niter)
    
    # evaluate the model on the training dataset
    probs = model.eval(X)
    Y = np.argmax(probs, axis=1)
    
    # report performance
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print ("accuracy recall precision:\n", accuracy, recall, precision)

    plt.figure()

    decfun = tf_deep.tf_deep_decfun(model)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=1, width=1000, height=1000)
    data.graph_data(X, Y_, Y)

    plt.show()

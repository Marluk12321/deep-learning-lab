import numpy as np
import matplotlib.pyplot as plt
#import binlogreg



class Random2DGaussian:

    def __init__(self):
        minx = 0; maxx = 10
        miny = 0; maxy = 10

        self.mu = np.random.random_sample(2)
        self.mu[0] *= maxx - minx
        self.mu[1] *= maxy - miny

        D = np.zeros((2,2))
        D[0, 0] = (np.random.random_sample() * (maxx - minx)/5) ** 2
        D[1, 1] = (np.random.random_sample() * (maxy - miny)/5) ** 2
        angle = np.random.random_integers(-90, 90)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        self.sigma = R.T.dot(D).dot(R)

    def get_sample(self, n):
        return np.random.multivariate_normal(self.mu, self.sigma, n)

if __name__ == "__main__":
    np.random.seed(100)
    G = Random2DGaussian()
    X = G.get_sample(100)
    plt.title("Random2DGaussian test")
    plt.scatter(X[:,0], X[:,1])
    plt.show()



def sample_gauss_2d(C, N):
    G = [Random2DGaussian() for _ in range(C)]
    Y = np.random.random_integers(0, C-1, N)
    X = np.array([G[y].get_sample(1)[0] for y in Y])
    return X, Y

def eval_perf_binary(Y, Y_):
    ones = np.ones((Y.shape))

    tp = np.logical_and(Y_==1, Y==1)
    tn = np.logical_and(Y_==0, Y==0)
    fp = np.logical_and(Y_==0, Y==1)
    fn = np.logical_and(Y_==1, Y==0)

    TP = np.sum(ones[tp])
    TN = np.sum(ones[tn])
    FP = np.sum(ones[fp])
    FN = np.sum(ones[fn])

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return accuracy, precision, recall

def eval_AP(Y_):
    N = Y_.shape[0]

    result = 0.0
    for i in range(N):
        TP = np.sum(Y_[i:])
        FP = N - i - TP
        result += Y_[i] * TP / (TP + FP)

    return result / np.sum(Y_)



def graph_data(X, Y_, Y, special=[]):
    '''
    X  ... podatci (np.array dimenzija NxD)
    Y_ ... tocni indeksi razreda podataka (Nx1)
    Y  ... predvidjeni indeksi razreda podataka (Nx1)
    '''
    C = np.max(Y_) + 1
    M = np.zeros((C, C))

    ones = np.ones((Y.shape))
    for i in range(C):
        for j in range(C):
            mask = np.logical_and(Y_==i, Y==j)
            M[i, j] = np.sum(ones[mask])

    colors = np.linspace(0.2, 1.0, num=C)
    for i in range(C):
        tp = np.logical_and(Y_==i,                Y==i)
        fn = np.logical_and(Y_==i, np.logical_not(Y==i))
        color = str(colors[i])
        plt.scatter(X[tp,0], X[tp,1], color=color, edgecolors="black", marker="o")
        plt.scatter(X[fn,0], X[fn,1], color=color, edgecolors="black", marker="s")
    
    for i in special:
        color = str(colors[ Y_[i] ])
        marker = "o" if Y_[i]==Y[i] else "s"
        plt.scatter(X[i,0], X[i,1], color=color, edgecolors="black", marker=marker, s=40)

def myDummyDecision(X):
    scores = X[:,0] + X[:,1] - 5
    return scores

if __name__ == "__main__":
    np.random.seed(100)
    plt.figure()
    plt.title("graph_data test")

    # get the training dataset
    X,Y_ = sample_gauss_2d(2, 100)

    # get the class predictions
    Y = myDummyDecision(X) > 0.5

    # graph the data points
    graph_data(X, Y_, Y)

    # show the results
    plt.show()



def  graph_surface(fun, rect, offset, width, height):
    '''
    fun    ... decizijska funkcija (Nx2)->(Nx1)
    rect   ... željena domena prikaza zadana kao:
                ([x_min,y_min], [x_max,y_max])
    offset ... "nulta" vrijednost decizijske funkcije na koju 
                je potrebno poravnati središte palete boja;
                tipično imamo:
                offset = 0.5 za probabilističke modele 
                    (npr. logistička regresija)
                offset = 0 za modele koji ne spljošćuju
                    klasifikacijske rezultate (npr. SVM)
    width,height ... rezolucija koordinatne mreže
    '''
    ([x_min,y_min], [x_max,y_max]) = rect
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    xx, yy = np.meshgrid(x, y)
    
    xx_flat = xx.reshape((width*height,))
    yy_flat = yy.reshape((width*height,))

    X = np.stack((xx_flat, yy_flat), axis=1)
    Y = fun(X).reshape(xx.shape)

    y_range = np.abs(np.max(Y) - np.min(Y))
    plt.pcolormesh(xx, yy, Y, vmin=offset-y_range/2, vmax=offset+y_range/2)
    plt.contour(xx, yy, Y, levels=[offset])


if __name__ == "__main__":
    np.random.seed(100)
    plt.figure()
    plt.title("graph_surface test")

    # get the training dataset
    X,Y_ = sample_gauss_2d(2, 100)

    # get the class predictions
    Y = myDummyDecision(X) > 0.5

    # graph the data points
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(myDummyDecision, bbox, offset=0.5, width=100, height=100)
    graph_data(X, Y_, Y)

    # show the results
    plt.show()


'''
if __name__ == "__main__":
    np.random.seed(100)
    plt.figure()
    plt.title("binlogreg graph_surface test")

    # instantiate the dataset
    X,Y_ = sample_gauss_2d(2, 100)

    # train the logistic regression model
    w, b = binlogreg.binlogreg_train(X, Y_, param_niter=500, param_delta=0.1)

    # evaluate the model on the train set
    probs = binlogreg.binlogreg_classify(X, w, b)

    # recover the predicted classes Y
    Y = np.around(probs)

    # evaluate and print performance measures
    accuracy, recall, precision = eval_perf_binary(Y, Y_)
    AP = eval_AP(Y_[probs.argsort()])
    print (accuracy, recall, precision, AP)

    # graph the decision surface
    decfun = binlogreg.binlogreg_decfun(w, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(decfun, bbox, offset=0.5, width=100, height=100)

    # graph the data points
    graph_data(X, Y_, Y)

    # show the results
    plt.show()
'''


def eval_perf_multi(Y, Y_):
    C = np.max(Y_) + 1
    M = np.zeros((C, C))

    ones = np.ones((Y.shape))
    for i in range(C):
        for j in range(C):
            mask = np.logical_and(Y_==i, Y==j)
            M[i, j] = np.sum(ones[mask])

    I = np.identity(C)
    diagonal = M[I==1]

    accuracy = np.sum(diagonal) / np.sum(M)
    precision = diagonal / np.sum(M, axis=0)
    recall = diagonal / np.sum(M, axis=1)

    '''
    for i in range(C):
        precision[i] = M[i,i] / np.sum(M[:,i])
        recall[i] = M[i,i] / np.sum(M[i,:])
    '''

    return accuracy, precision, recall



def sample_gmm_2d(K, C, N):
    '''
    Ulaz:
    K  ... broj slučajnih bivarijatnih Gaussovih razdioba
    C  ... broj razreda
    N  ... broj primjera za svaku razdiobu

    Izlaz:
    X  ... podatci u matrici [K·N x 2 ]
    Y_ ... indeksi razreda podataka [K·N]
    '''
    generators = [Random2DGaussian() for _ in range(K)]
    G = [generators[np.random.randint(0, K-1)] for _ in range(C)]
    Y = np.random.random_integers(0, C-1, N*K)
    X = np.array([G[y].get_sample(1)[0] for y in Y])
    return X, Y

if __name__ == "__main__":
    np.random.seed(100)

    K = 5
    C = 2
    N = 30

    # get the training dataset
    X, Y_ = sample_gmm_2d(K, C, N)

    # get the class predictions
    Y = myDummyDecision(X) > 0.5

    plt.figure()
    plt.title("sample_gmm_2d test")

    # graph the data points
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(myDummyDecision, bbox, offset=0.5, width=100, height=100)
    graph_data(X, Y_, Y)

    # show the results
    plt.show()

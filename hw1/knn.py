import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy import stats
get_ipython().magic('matplotlib inline')


# Get Data

X = []
y = []
trdata = pd.read_csv('mnist_train.csv', sep=',', engine='python')
tstdata = pd.read_csv('mnist_test.csv', sep=',', engine='python')

X = trdata.as_matrix()
X = X.astype(np.float)
y = X[:,0]
X = X[:,1:]
print(X.shape, y.shape)

# test data
Xt = tstdata.as_matrix()
Xt = Xt.astype(np.float)
yt = Xt[:,0]
Xt = Xt[:,1:]
print(Xt.shape, yt.shape)


# Downsample Data
batchsize = 3000

usedinds = np.random.randint(X.shape[0], size = batchsize)

X = X[usedinds,:]
y = y[usedinds]

usedinds = np.random.randint(Xt.shape[0], size = batchsize)

Xt = Xt[usedinds,:]
yt = yt[usedinds]

# Define KNN
def knn(k):
    prediction = np.zeros(X.shape[0])
    for i in range(Xt.shape[0]):
        point = Xt[i]
        dists = np.linalg.norm(X - point[:,np.newaxis].T, axis=1)
        smallestdistindices = np.argpartition(dists, k)[k]
        prediction[i] = stats.mode(y[smallestdistindices])[0]
            
    return prediction  


# Define accuracy function
def accuracy(ypred, yreal):
    correct = 0
    for pred,real in zip(ypred, yreal):
        correct += int(pred == real)
    return correct / y.shape[0]


# Get accuracy at desired ks
print(accuracy(knn(1), yt), accuracy(knn(5), yt), accuracy(knn(10), yt))


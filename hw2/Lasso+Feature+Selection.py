import numpy as np
import csv
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
inv = np.linalg.inv
plt.style.use('ggplot')

# import data
with open('../hw1/online_news_popularity.csv', 'r') as f:
    reader = csv.reader(f)
    a = list(reader)
    labels = a[0]
    data = np.array(a[1:])
data.shape

# convert the data into a convenient array
data = data[:,1:]
arrToFloat = np.vectorize(float)
data = arrToFloat(data)
shares = np.log(data[:,-1])

# split data into training and test data
dataRatio = .1
trainingrows = int(data.shape[0]*dataRatio)
trdata = data[0:trainingrows, :]
trshares = shares[0:trainingrows]
tstdata = data[trainingrows:, :]
tstshares = shares[trainingrows:]

# rename and normalize data
X = trdata / trdata.sum(axis = 1)[:, np.newaxis]
y = trshares

# define learning parameters
wlrn = 2.5e-5
regconst = 10000000.
iters = 1000

# gradient function
def grad(weights, regconst):
    return 2 * X.T @ X @ weights - 2 * y.T @ X + regconst * np.sign(weights)

# threshold the gradient
def prox(weights, regconst):
    update = weights - wlrn * grad(weights, regconst)
    update[abs(update) < wlrn] = 0
    update[update > wlrn] -= wlrn
    update[update < wlrn] += wlrn
    return update

def objective(weights, regconst):
    error = X @ weights - y
    return (error.T @ error + regconst * sum(abs(weights)))/y.shape[0]

def grad_descent(regconst, iters = 100, verbose = False):
    # initialize weights randomly
    weights = np.random.randn(trdata.shape[1])
    
    # initialize weights with linear regression closed form
    # weights = np.linalg.inv(X.T @ X) @ X.T @ y
    
    regconst /= y.shape[0]
    
    objectives = []
    # run gradient descent
    for i in range(iters):
        # update weights
        weights = prox(weights, regconst)

        objectives += [objective(weights, regconst)]
        if i % (iters/10) == 0 and verbose:
            print(objectives[i]) 
    return weights, objectives

objectives = grad_descent(regconst, iters, True)[1]

# plot objective
plt.figure(figsize = (15, 10))
plt.subplot(211)
plt.plot(objectives[0:200])
plt.ylabel("Objective")
plt.title("Regularization Constant = " + str(regconst))         
plt.subplot(212)
plt.plot(list(range(200, len(objectives))), objectives[200:])
plt.ylabel("Objective")
plt.xlabel("Iterations")


# regularization path
iterations = 1000
num_regs = 100

optweights = np.zeros((num_regs, 60))
invregs = np.linspace(1e-9,.000001, num_regs)

i = 0
for invreg in invregs:
    # initialize weights randomly
    optweights[i] = grad_descent(1./invreg)[0]
    i += 1
print('done')

# plot regularization path
plt.figure(figsize = (15,10))
plt.plot(invregs, optweights)
plt.xlabel("Inverse Regularization Constant")
plt.ylabel("Optimal Weight")
plt.title("Regularization Path")


# Find most relevant features
means = np.mean(optweights, axis = 0)
relevant_features = []
for i in range(5):
    relevant_features += [np.argmax(abs(means))]
    means[relevant_features[i]] = 0
for r in relevant_features:
    print(labels[r])

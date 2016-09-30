# imports and plotting
import numpy as np
import csv
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# closed form for part b
x = np.array([0,2,3,4])
y = [1,3,6,8]
m = 62/35
b = 18/35

m * x + b

# plot for part c
plt.scatter(x,y)
plt.plot(x, m * x +b )

# test data for part d
newX = np.linspace(0,99,100)
newY = m * newX + b + 20*np.random.randn(100)
plt.scatter(newX,newY)
newX = newX.reshape(100,1)
newX = np.hstack((np.ones((100,1)),newX))
print(newX.shape)
fit = np.linalg.inv(newX.T @ newX) @ newX.T @ newY
plt.plot(newX[:,1],newX @ fit)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def knn(x0, x, y, K=3):
    d = np.abs(x - x0)        
    idx = np.argsort(d)[:K]
    return np.mean(y[idx])

# Simulate data
n = 100
np.random.seed(123)
x1 = np.random.normal(3,2,size=n)
x2 = np.random.normal(3,1,size=n)
e = np.random.normal(0,0.1,size=n)
y = 1 + x1 - 0.6*x1**2 + 2*x2 + 0.6*x2**2 + e

# GAM
b0 = y.mean()
yd = y-b0 # Demean y
fig, ax = plt.subplots(1,2)
ax[0].scatter(x1,y)
ax[1].scatter(x2,y)
K=20


# First iteration
f1 = b0 + np.array([knn(x0,x1,yd,K) for x0 in x1])
# Estimate f1 and add b0
o = np.argsort(x1)
ax[0].plot(x1[o],f1[o],color="red")

res = y-f1 # Compute residuals
res = res - res.mean() # Demean residuals
f2 = b0 + np.array([knn(x0,x2,res,K) for x0 in x2])
# Estimate f2 and add b0
o = np.argsort(x2)
ax[1].plot(x2[o],f2[o],color="red")

# Try with more interations to confirm that it converges
for i in range(10):
    res = y-f2
    res = res - res.mean()
    f1 = b0 + np.array([knn(x0,x1,res,K) for x0 in x1])
    o = np.argsort(x1)
    ax[0].plot(x1[o],f1[o],color="magenta")    
    res = y-f1
    res = res - res.mean()
    f2 = b0 + np.array([knn(x0,x2,res,K) for x0 in x2])
    print(f1[10:12].round(3))
    # Print f1 and f2 for a few values in each iteration
    o = np.argsort(x2)
    ax[1].plot(x2[o],f2[o],color="magenta")

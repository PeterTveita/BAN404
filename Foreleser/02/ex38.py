
# Exercise 3.8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (summarize,ModelSpec as MS)

# (a)
# Load data
Auto = load_data("Auto")

# First look at data
Auto.head()
Auto.dtypes
plt.scatter(Auto["horsepower"], Auto["mpg"])
plt.xlabel("horsepower")
plt.ylabel("mpg")
plt.show()

# (a) Linear regression
#X = Auto[["horsepower"]].values
#y = Auto["mpg"].values
#X_sm = sm.add_constant(X)
design = MS(['horsepower'])
design = design.fit(Auto)
X = design.transform(Auto)
y = Auto["mpg"]
model = sm.OLS(y, X)
results = model.fit()
results.summary()
summarize(results) # Alternative using the ISLP.models function summarize

# (i)   The null hypothesis that horsepower is not associated with mpg is rejected
#       with a p-value smaller than 2e-16
# (ii)  The strength of the relationship, measured as R^2, is 0.60.
# (iii) The relationship is negative. An increase with one horsepower is associated
#       with a decrease in fuel usage of -0.16 gallon
# (iv) Prediction at horsepower = 98
X_new = pd.DataFrame({'horsepower':[98]})
X_new = design.transform(X_new)

pred1 = results.predict(X_new)
pred1

# C# Confidence interval. Closeness of prediction and f(X)
pred2 = results.get_prediction(X_new).conf_int(alpha = 0.05)
pred2

# Prediction interval. Closeness of prediction and Y
pred3 = results.get_prediction(X_new).conf_int(obs=True,alpha = 0.05)
pred3

# (b) Scatter + regression line
ax = Auto.plot.scatter('horsepower','mpg')
ax.axline(xy1=[0,results.params[0]],slope=results.params[1],color='red')

# (c) Diagnostic plots (residuals and QQ-plot)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].scatter(results.fittedvalues, results.resid)
axs[0].axhline(0, color="black")
axs[0].set_xlabel("Fitted values")
axs[0].set_ylabel("Residuals")
sm.qqplot(results.resid, line="45", loc=np.mean(results.resid),scale=np.std(results.resid),ax=axs[1])
# Systematic pattern in plot of residuals on predictions
# Heteroscedasticity and positive residuals for large and small 
# predictions. Nonlinear relationship? This might cause significance levels
# and levels of confidence intervals not being as they are supposed to be, e.g.,
# 5% or 95%.
# Some deviation from normality. Only a problem for prediction intervals.


# Using KNN
plt.figure()
def knn(x0, x, y, K=3):
    d = np.abs(x - x0)
    o = np.argsort(d)[:K]
    return y[o].mean()

x = Auto["horsepower"].values
y = Auto["mpg"].values
n = len(y)

o = np.argsort(x)
xo = x[o]
yo = y[o]

yp = np.zeros(n)
for i in range(n):
    yp[i] = knn(xo[i], xo, yo, K=20)

plt.scatter(x, y)
plt.plot(xo, yp, color="blue")
plt.show()

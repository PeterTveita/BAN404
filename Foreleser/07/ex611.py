# Read libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ISLP import load_data
import sklearn.linear_model as skl
import statsmodels.api as sm

# Load data
boston = load_data("Boston")
y = boston['crim']
X = boston.drop(columns='crim')

# Train/test
np.random.seed(1234)
n = len(y)
ntrain = n // 2
ind = np.random.choice(np.arange(n), size=ntrain, replace=False)
mask = np.ones(n,dtype = bool)
mask[ind] = False
Xtrain = X.iloc[ind]
ytrain = y.iloc[ind]
Xtest = X.iloc[mask]
ytest = y.iloc[mask]

# OLS
Xtrain_with_const = sm.add_constant(Xtrain)
ols = sm.OLS(ytrain, Xtrain_with_const).fit()
print(ols.summary())
Xtest_with_const = sm.add_constant(Xtest)
pred_ols = ols.predict(Xtest_with_const)
olserr = ytest - pred_ols
MSEols = (olserr**2).mean()

# Ridge
alphas = np.logspace(-4, 4, 200)
ridge_cv = skl.RidgeCV(alphas=alphas)
ridge_cv.fit(Xtrain, ytrain)
alphamin = ridge_cv.alpha_
print("alpha.min =", round(alphamin,2))
ridgemin = skl.Ridge(alpha=alphamin)
ridgemin.fit(Xtrain, ytrain)

# Compare coefficients
ols_coef = ols.params
ridge_coef = np.concatenate(([ridgemin.intercept_], ridgemin.coef_))
coef_comparison = np.column_stack((ols_coef, ridge_coef))
coef_comparison = pd.DataFrame(coef_comparison,columns=['OLS','Ridge'])
print(coef_comparison)

# Lasso
alphas = np.logspace(-4, 4, 200)
lasso_cv = skl.LassoCV(alphas = alphas)
lasso_cv.fit(Xtrain, ytrain)

lassoalpha = lasso_cv.alpha_
print("alpha.min (lasso) =", lassoalpha.round(2))

lasso = skl.Lasso(alpha=lassoalpha)
lasso.fit(Xtrain, ytrain)
lasso_coef = np.concatenate(([lasso.intercept_], lasso.coef_))
coef_comparison = np.column_stack((ols_coef, ridge_coef, lasso_coef))
coef_comparison = pd.DataFrame(coef_comparison,columns=['OLS','Ridge','Lasso'])
print(coef_comparison)

# Comparison on test data
pred_ridge = ridge_cv.predict(Xtest)
ridgeerr = ytest - pred_ridge
MSEridge = (ridgeerr**2).mean()
pred_lasso = lasso.predict(Xtest)
lassoerr = ytest - pred_lasso
MSElasso = (lassoerr**2).mean()

print(round(MSEols,3),round(MSEridge,3),round(MSElasso,3))

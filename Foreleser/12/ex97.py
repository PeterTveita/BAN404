import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from ISLP.svm import plot as plot_svm
from ISLP import confusion_table

from ISLP import load_data

# =========================
# Read data  (ISLR Auto)
# =========================

Auto = load_data("Auto")
print(Auto.describe())

# (a) Create 0-1 variable for mpg above median
Auto["high"] = (Auto["mpg"] > Auto["mpg"].median()).astype(int)

# Remove mpg (like Auto = Auto[,-1] in R)
Auto = Auto.drop(columns=["mpg"])

# =========================
# (b) Linear SVC
# =========================

X = Auto[["horsepower", "weight", "year"]]
y = Auto["high"]

svm_linear = SVC(kernel="linear", C=10)
svm_linear.fit(X, y)

# Plots
fig, ax = plt.subplots(2, 2)

plot_svm(X, y, svm_linear, ax=ax[0, 0], features=(0, 1))
ax[0, 0].set_xlabel(X.columns[0])
ax[0, 0].set_ylabel(X.columns[1])

plot_svm(X, y, svm_linear, ax=ax[0, 1], features=(0, 2))
ax[0, 1].set_xlabel(X.columns[0])
ax[0, 1].set_ylabel(X.columns[2])

plot_svm(X, y, svm_linear, ax=ax[1, 0], features=(1, 2))
ax[1, 0].set_xlabel(X.columns[1])
ax[1, 0].set_ylabel(X.columns[2])

plt.tight_layout()
plt.show()

# Find optimal C (inversely related to K)
param_grid = {"C": [0.01, 0.1, 1, 10, 100, 1000, 5000]}
tuneC = GridSearchCV(SVC(kernel="linear"),param_grid, cv=2, scoring="accuracy")
tuneC.fit(X, y)

print("Best C (linear):", tuneC.best_params_)
bestmod = tuneC.best_estimator_

# =========================
# (c) Polynomial and radial kernels
# =========================

# polynomial tuning
param_poly = {
    "C": [0.01, 0.1, 1, 10, 100, 1000, 5000,10000],
    "degree": [1, 2, 3, 4],
}
tuneCd = GridSearchCV(SVC(kernel="poly"), param_poly, cv=5)
tuneCd.fit(X, y)

print("Best poly:", tuneCd.best_params_)
bestpoly = tuneCd.best_estimator_

# radial tuning
param_rbf = {
    "C": [0.01, 0.1, 1, 10, 100, 1000],
    "gamma": [0.5, 1, 2, 3, 4],
}
tuneCg = GridSearchCV(SVC(kernel="rbf"), param_rbf, cv=5)
tuneCg.fit(X, y)

print("Best radial:", tuneCg.best_params_)
bestradial = tuneCg.best_estimator_

# =========================
# (d) Compare models visually
# =========================
X.columns
fig, ax = plt.subplots(2, 2)
plot_svm(X, y, bestmod, features=(0, 1), ax = ax[0,0])
plot_svm(X, y, bestpoly, features=(0, 1), ax = ax[0,1])
plot_svm(X, y, bestradial, features=(0, 1), ax = ax[1,0])
plt.tight_layout()
plt.show


fig, ax = plt.subplots(2, 2)

plot_svm(X, y, bestpoly, ax=ax[0, 0], features=(0, 1))
ax[0, 0].set_xlabel(X.columns[0])
ax[0, 0].set_ylabel(X.columns[1])

plot_svm(X, y, bestpoly, ax=ax[0, 1], features=(0, 2))
ax[0, 1].set_xlabel(X.columns[0])
ax[0, 1].set_ylabel(X.columns[2])

plot_svm(X, y, bestpoly, ax=ax[1, 0], features=(1, 2))
ax[1, 0].set_xlabel(X.columns[1])
ax[1, 0].set_ylabel(X.columns[2])

plt.tight_layout()
plt.show()

# Radial
fig, ax = plt.subplots(2, 2)

plot_svm(X, y, bestpoly, ax=ax[0, 0], features=(0, 1))
ax[0, 0].set_xlabel(X.columns[0])
ax[0, 0].set_ylabel(X.columns[1])

plot_svm(X, y, bestpoly, ax=ax[0, 1], features=(0, 2))
ax[0, 1].set_xlabel(X.columns[0])
ax[0, 1].set_ylabel(X.columns[2])

plot_svm(X, y, bestpoly, ax=ax[1, 0], features=(1, 2))
ax[1, 0].set_xlabel(X.columns[1])
ax[1, 0].set_ylabel(X.columns[2])

plt.tight_layout()
plt.show()

badmodel = SVC(kernel="linear", C=0.01)
badmodel.fit(X, y)

plt.scatter(Auto["year"], Auto["weight"], c=badmodel.predict(X))
plt.title("Bad linear model classification")
plt.show()

plt.scatter(Auto["year"], Auto["weight"], c=bestmod.predict(X))
plt.title("Best linear model classification")
plt.show()

plt.scatter(Auto["year"], Auto["weight"], c=bestpoly.predict(X))
plt.title("Best polynomial model classification")
plt.show()

plt.scatter(Auto["year"], Auto["weight"], c=bestradial.predict(X))
plt.title("Best radial model classification")
plt.show()

plt.scatter(Auto["horsepower"], Auto["weight"], c=bestradial.predict(X))
plt.title("Radial model: horsepower vs weight")
plt.show()

# =========================
# Training confusion matrices
# =========================

predbest = bestmod.predict(X)
print("Linear best\n", confusion_matrix(y, predbest))

predbad = badmodel.predict(X)
print("Linear bad\n", confusion_matrix(y, predbad))

predpoly = bestpoly.predict(X)
print("Polynomial\n", confusion_matrix(y, predpoly))

predradial = bestradial.predict(X)
print("Radial\n", confusion_matrix(y, predradial))

# Load libraries
import numpy as np
import pandas as pd
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from sklearn.model_selection import (train_test_split,GridSearchCV)
import statsmodels.api as sm
from ISLP.models import summarize

# Function to compute MAE
def MAE(ypred,y):
    score = np.mean(np.abs(y-ypred))
    return score


# Read data
import pandas as pd
ames = pd.read_csv("AmesHousing.csv")
# The subset we will work with
ames_small = ames[["Gr Liv Area","Total Bsmt SF",
        "Lot Area","Year Built",
        "Overall Qual","Overall Cond",
        "Full Bath","Garage Cars",
        "Central Air","Neighborhood",
        "House Style","Bldg Type","SalePrice"]]

# Create dummies for categorical variables
ames_small = pd.get_dummies(ames_small,drop_first=True)
# Make sure that they are numerical
ames_small = ames_small.astype(float)
# Extract predictors and respons variable
X = ames_small.drop(columns='SalePrice')
Y = ames_small['SalePrice']
# Check for missing values
X.isna().sum().sort_values(ascending=False).head(10)
# fill NaNs with 0 for structural absence
X = X.fillna(0)

(X_train,X_test,Y_train,Y_test) = train_test_split(X,Y,
            test_size =1/2,random_state =1)

# Linear regression
X_train_const = sm.add_constant(X_train)
reg = sm.OLS(Y_train,X_train_const).fit()
reg.summary().tables[1]
X_test_const = sm.add_constant(X_test)
y_pred = reg.predict(X_test_const)
print(MAE(y_pred,Y_test).round(1))

# Ridge regression
scaler = StandardScaler(with_mean=True , with_std=True)
alphas = np.logspace(-4, 4, 100)
ridge_cv = skl.RidgeCV(alphas=alphas)
X_train_stand = scaler.fit_transform(X_train)
ridge_cv.fit(X_train_stand, Y_train)
alphamin = ridge_cv.alpha_
print("alpha.min =", round(alphamin,2))
ridgemin = skl.Ridge(alpha=alphamin)
ridgemin.fit(X_train_stand, Y_train)
X_test_stand = scaler.fit_transform(X_test)
y_pred = ridgemin.predict(X_test_stand)
print(MAE(y_pred,Y_test).round(1))

# LASSO regression
alphas = np.logspace(-4, 4, 100)
lasso_cv = skl.LassoCV(alphas=alphas)
lasso_cv.fit(X_train_stand, Y_train)
alphamin = lasso_cv.alpha_
print("alpha.min =", round(alphamin,2))
lassomin = skl.Lasso(alpha=alphamin)
lassomin.fit(X_train_stand, Y_train)
y_pred = lassomin.predict(X_test_stand)
print(MAE(y_pred,Y_test).round(1))

# Bagged tree
from sklearn.ensemble import RandomForestRegressor
bagging = RandomForestRegressor(n_estimators=500,
max_features=X_train.shape[1], random_state=123)
bagging.fit(X_train, Y_train)
y_pred = bagging.predict(X_test)
print(MAE(y_pred,Y_test).round(1))

# Random forest
rf = RandomForestRegressor(n_estimators=500,
max_features=3, random_state=123)
rf.fit(X_train, Y_train)
y_pred = rf.predict(X_test)
print(MAE(y_pred,Y_test).round(1))

# Boosted tree
from sklearn.ensemble import GradientBoostingRegressor
boost = GradientBoostingRegressor(n_estimators=1000,
learning_rate=0.01, max_depth=2, random_state=123)
boost.fit(X_train, Y_train)
y_pred = boost.predict(X_test)
print(MAE(y_pred,Y_test).round(1))

# Single layered network
nn = MLPRegressor(
    hidden_layer_sizes=(15,),
    activation="relu",
    max_iter=100000,
    random_state=0
)
nn.fit(X_train, Y_train)
y_pred = nn.predict(X_test)
print(MAE(y_pred,Y_test).round(1))

# Same with standardized X-variables
nn.fit(X_train_stand, Y_train)
y_pred = nn.predict(X_test_stand)
print(MAE(y_pred,Y_test).round(1))

# Multi layered network
dnn = MLPRegressor(
    hidden_layer_sizes=(15,8),
    activation="relu",
    max_iter=100000,
    random_state=0
)

dnn.fit(X_train_stand, Y_train)
y_pred = dnn.predict(X_test_stand)
print(MAE(y_pred,Y_test).round(1))

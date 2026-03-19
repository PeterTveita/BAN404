# (a)

# Read libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ISLP import load_data

# First look at data
Weekly = load_data('Weekly')
Weekly.describe(include = 'all').round(3)
print(Weekly.head())
print(Weekly.columns)

# Time series plot of Today
fig, ax = plt.subplots()
ax.plot(Weekly["Today"])
ax.set_xlabel("Week")
ax.set_ylabel("Today")
ax.set_title("Weekly Market Returns")

# Distribution of Direction
print(Weekly["Direction"].value_counts())

# Mean of Volume
print("Mean Volume:", Weekly["Volume"].mean())

# Mean Volume by Direction
print(Weekly.groupby("Direction")["Volume"].mean())

# T-test of difference in volume for up and down
from scipy import stats
x = Weekly.Volume[Weekly.Direction=='Up']
y = Weekly.Volume[Weekly.Direction=='Down']
nx = len(x)
ny = len(y)
tval = (x.mean()-y.mean())/np.sqrt(x.var()/nx + y.var()/ny)
pval = 2*stats.norm.cdf(-np.abs(tval))
print("tval =",f"{tval:.3f}")
print("pval =",f"{pval:.3f}")

# We can not reject the hypotesis
# $H_0: \mu_x = \mu_y$ against the alternative
# $H_1: \mu_x \neq \mu_y$
# at any conventional significance levels.


# (b)
import statsmodels.formula.api as smf
from ISLP.models import summarize
Weekly['y'] = (Weekly['Direction'] == 'Up').astype(int)
model1 = smf.logit('y ~ Volume + Lag1 + Lag2 + Lag3 + Lag4 + Lag5',data=Weekly)
m1 = model1.fit()
summarize(m1)

# (c)
prob = m1.predict()
pred = prob > 0.5
confusion_table = pd.crosstab(Weekly['y'],
 pred, rownames=['Actual'], colnames=['Predicted'])
print(confusion_table)
accuracy = np.mean(Weekly['y']==pred.astype(int))
print(accuracy.round(2))

## (d)
# Test / training data
np.random.seed(431)
n = len(Weekly)
ind = Weekly['Year'] < 2009
train = Weekly[ind]
test = Weekly[~ind]
m1_train = smf.logit('y ~ Volume + Lag1 + Lag2 + Lag3 + Lag4 + Lag5',data=train).fit()
prob_test = m1_train.predict(test)
pred_test = prob_test > 0.5
confusion_table = pd.crosstab(test['y'],
 pred_test, rownames=['Actual'], colnames=['Predicted'])
print(confusion_table)
accuracy = np.mean(test['y']==pred_test.astype(int))
print("accuracy =",accuracy.round(2))

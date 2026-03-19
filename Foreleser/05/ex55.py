
# Read libraries
import numpy as np
import pandas as pd
from ISLP import load_data
import statsmodels.api as sm
import statsmodels.formula.api as smf

# (a)
Default = load_data("Default")
logreg1 = smf.glm('default ~ income + balance', data=Default, family=sm.families.Binomial()).fit()
print(logreg1.summary())

# (b)
n = len(Default)
ntrain = n // 2
ind = np.random.choice(n, size=ntrain, replace=False)

train = Default.iloc[ind]
test = Default.drop(ind)

# Fit logistic regression model on training data
logreg2 = smf.glm('default ~ income + balance', data=train, family=sm.families.Binomial()).fit()
# Can also use smf.logit but the default has to be decoded as a numerical
print(logreg2.summary())

# Predict probabilities on test data
prob2 = logreg2.predict(test)

# Convert probabilities to class predictions with threshold 0.5
pred2 = prob2 > 0.5

# Confusion matrix: rows = true labels, columns = predicted labels
conf_mat = pd.crosstab(test['default'], ~pred2)
print(conf_mat)

# Validation accuracy
accuracy = np.trace(conf_mat) / ntrain
print(f'Validation accuracy: {accuracy:.4f}')

# (c)
# The error is around 0.97 each time but varies since the
# training/test division is not identical each time

# (d)

# Fit logistic regression model on training data
logreg3 = smf.glm('default ~ income + balance + student', data=train, family=sm.families.Binomial()).fit()
# Can also use smf.logit but the default has to be decoded as a numerical
print(logreg3.summary())

# Predict probabilities on test data
prob3 = logreg3.predict(test)

# Convert probabilities to class predictions with threshold 0.5
pred3 = prob3 > 0.5

# Confusion matrix: rows = true labels, columns = predicted labels
conf_mat = pd.crosstab(test['default'], ~pred3)
print(conf_mat)

# Validation accuracy
accuracy = np.trace(conf_mat) / ntrain
print(f'Validation accuracy: {accuracy:.4f}')


# Extra task: LOOCV


nsmall = 500

# Sample nsmall rows without replacement
small = Default.sample(nsmall, random_state=42).reset_index(drop=True)

pred4 = np.zeros(nsmall, dtype=bool)

for i in range(nsmall):
    # Leave one out
    training = small.drop(i)
    test = small.iloc[i:i+1]

    # Fit logistic regression on training data
    logreg4 = smf.glm('default ~ income + balance + student', data=training, family=sm.families.Binomial()).fit()

    # Predict probability on left-out test data
    prob4 = logreg4.predict(test).values[0]

    # Classify with 0.5 threshold
    pred4[i] = prob4 > 0.5

# Confusion matrix and accuracy
conf_mat = pd.crosstab(small['default'], ~pred4)
accuracy = np.trace(conf_mat) / nsmall

print("Confusion matrix:\n", conf_mat)
print(f"LOOCV accuracy: {accuracy:.4f}")


# Check the function LeaveOneOut in the library sklearn.model_selection
# to see how this can be done more efficiently
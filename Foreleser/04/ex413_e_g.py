
# (e)
# Read libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ISLP import load_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

# Features and target from train and test DataFrames
X_train = train[['Lag2']].values  # 2D array expected by sklearn
y_train = train['Direction'].values
X_test = test[['Lag2']].values
y_test = test['Direction'].values

# Fit LDA model
lda1 = LinearDiscriminantAnalysis()
lda1.fit(X_train, y_train)

# Predict classes and posterior probabilities on test set
pred5_class = lda1.predict(X_test)
pred5_posterior = lda1.predict_proba(X_test)
pred5_class = pred5_posterior[:,1] > 0.54
pred5_class = np.where(pred5_class,'Up','Down')

# Confusion matrix
conf_matrix = pd.crosstab(y_test,pred5_class)
print(conf_matrix)

# Accuracy
accuracy = (pred5_class == y_test).mean()
print(f"Accuracy: {accuracy:.4f}")

# Class-wise proportions 
prop_table = pd.crosstab(y_test,pred5_class,
normalize='index')
prop_table

# (f)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Fit QDA model
qda1 = QuadraticDiscriminantAnalysis()
qda1.fit(X_train, y_train)

# Predict classes and posterior probabilities on test set
pred6_class = qda1.predict(X_test)
pred6_posterior = qda1.predict_proba(X_test)
pred6_class = pred6_posterior[:,1] > 0.55
pred6_class = np.where(pred6_class,'Up','Down')

# Confusion matrix
conf_matrix = pd.crosstab(y_test,pred6_class)
print(conf_matrix)

# Accuracy
accuracy = (pred6_class == y_test).mean()
print(f"Accuracy: {accuracy:.4f}")

# Class-wise proportions 
prop_table = pd.crosstab(y_test,pred6_class,
normalize='index')
prop_table

# (g)
from sklearn.neighbors import KNeighborsClassifier

# Instantiate KNN with k=1
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict on test set
pred7 = knn.predict(X_test)
prob7 = knn.predict_proba(X_test)
pred7 = prob7[:,1] > 0.5
pred7 = np.where(pred7,'Up','Down')

# Confusion matrix
conf_matrix = pd.crosstab(y_test,pred7)
conf_matrix

# Accuracy
accuracy = (pred7 == y_test).mean()
print(f"Accuracy: {accuracy:.4f}")

# Proportion table by true class (margin=1)
prop_table = pd.crosstab(y_test,pred7,
normalize='index')
prop_table



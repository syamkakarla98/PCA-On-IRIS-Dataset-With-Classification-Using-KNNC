import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn import decomposition
from sklearn import datasets

# load dataset into Pandas DataFrame
df = pd.read_csv("D:\Python_programs\ML\iris_after_pca.csv")
#df.to_csv('iris.csv')
from sklearn.preprocessing import StandardScaler

features = ['PC-1', 'PC-2']
# Separating out the features
X = df.loc[:, features].values
# Separating out the target
Y = df.loc[:,['target']].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
 X, Y, test_size = 0.3, random_state = 100)
y_train=y_train.ravel()
y_test=y_test.ravel()
#classifier.fit(X_train, y_train.squeeze())

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)
Yhat = model.predict(X_test)

from sklearn import metrics
#acc = metrics.accuracy_score(Yhat, y_test)
#print(acc)
print('*'*11,'Accuracy of IRIS Dataset after PCA','*'*11,'\n')
for K in range(25):
 K_value = K+1
 neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')
 neigh.fit(X_train, y_train) 
 y_pred = neigh.predict(X_test)
 print ("Accuracy is ", metrics.accuracy_score(y_test,y_pred)*100,"% for K-Value:",K_value)


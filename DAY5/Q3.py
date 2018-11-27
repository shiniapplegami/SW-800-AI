from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("ex3.txt",delimiter=",")
#data=data.as_matrix()

X1=data.iloc[:,[0]].values
X2=data.iloc[:,[1]].values
ad=data.iloc[:,[2]].values

print(ad)

#Plot and show

plt.scatter(X1, X2, c=ad)
plt.show()

X=data.iloc[:,:-1].values
y=data.iloc[:, -1].values
print(X)
print(y)

#Splitting the data

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.40)

#Using KNN

knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

print(y_pred)

#Plot and show

plt.scatter(y_test,y_pred)
plt.show()

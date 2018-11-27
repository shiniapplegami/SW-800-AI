
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import  train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("ex1.txt",delimiter=",")
#data=data.as_matrix()

X=data.iloc[:,:-1].values
y=data.iloc[:, -1].values
print(X)
print(y)

#Splitting the data

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)

#Using Linear Regression

lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

print(y_pred)

#Plot graph and show

plt.scatter(y_test,y_pred)
plt.show()





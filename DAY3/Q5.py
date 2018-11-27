from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading the data

data=pd.read_csv("city.csv")
print(data)
print (data.head())
data=data.as_matrix()

X=data[:, 1:6]
y=data[:,-2]

#Splitting the data

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#Using Linear Regression

lg=LinearRegression()
lg.fit(X_train, y_train)
p1=lg.predict(X_test)

print(p1)

#Plot the graph

plt.scatter(y_test,p1)
plt.show()


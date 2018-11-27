
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read Data

data=pd.read_csv("pimaindians.csv")
print (data.head())
data=data.as_matrix()

X=data[:, 0:8]
y=data[:,-1]

#Splitting the data

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#Using Logistic Regression

lg=LogisticRegression()
lg.fit(X_train, y_train)
p1=lg.predict(X_test)

print(p1)

#Plot the graphs

plt.scatter(y_test,p1)
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("loan.csv")
print (data.head())
data=data.as_matrix()

X=data[:,[1,2]]
y=data[:,-2]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

lr1=LinearRegression()
lr1.fit(X_train, y_train)
p1=lr1.predict(X_test)

print(p1)
plt.scatter(y_test,p1)
plt.show()

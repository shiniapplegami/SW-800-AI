from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("Advertising.csv")
print (data.head())
data=data.as_matrix()

X=data[:,[1,2,3]]
y=data[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

lr=LinearRegression()
lr.fit(X_train, y_train)
p=lr.predict(X_test) 

print(p)
plt.scatter(y_test,p)
plt.show()








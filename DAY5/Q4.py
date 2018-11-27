from sklearn.linear_model import LinearRegression
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  train_test_split
#from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing



data=pd.read_csv("titanic.csv",delimiter=",")
t=data.as_matrix()

#Fill the missing values

data.fillna(data.mean(), inplace=True)

#print(data.head(20))
print(data.isnull().sum())

#Converting into categorical terms

le=preprocessing.LabelEncoder()
X=t[:,4]
le.fit(X)
P=le.transform(X)

data.iloc[:,4]=P
print(data.head(20))
'''
for col in .values:
       # Encoding only categorical variables
	if data[col].dtypes=='object':
	    	le.fit(data[col].values)
       		data[col]=le.transform(data[col])
       		

#data=data.as_matrix()

X1=data.iloc[:,[0]].values
X2=data.iloc[:,[1]].values
ad=data.iloc[:,[2]].values

print(ad)
plt.scatter(X1, X2, c=ad)
plt.show()
'''
X=data.iloc[:,[2,5,6,7,9]].values
y=data.iloc[:, 1].values
print(X)
print(y)

#Splitting the data

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

#Linear Regression

lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

print(y_pred)

#Plot and show

plt.scatter(y_test,y_pred)
plt.show()

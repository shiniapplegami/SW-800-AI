from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

iris=load_iris()
print(iris.data)
print(iris.target)
knn=KNeighborsClassifier(n_neighbors=1)
X=iris.data
y=iris.target

knn.fit(X,y)

p=knn.predict([[3,8,12,19]])

print(p)

#---------------------------------------------------------------------------------------------------------------------------------------------

X_train,X_test,y_train,y_test=train_test_split(X,y)
print(X_train.shape)
print(X_test.shape)

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

p=knn.predict(X_test)
print(confusion_matrix(y_test,p))
print(accuracy_score(y_test,p))

print(p)

#---------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

train1=pd.read_csv("/home/ai7/Desktop/common/ML/Day2/Questions/Immunotherapy.csv", delimiter=",")
test1=pd.read_csv("/home/ai7/Desktop/common/ML/Day2/Questions/Immunotherapy.csv", delimiter=",")
print(train1)
print(test1)

X_train,X_test,y_train,y_test=train_test_split(train1,test1)

train=train1.as_matrix()
test=test1.as_matrix()
print(train.shape)
print(test.shape)

iknn=KNeighborsClassifier(n_neighbors=1)
iknn.fit(train [:,0:6], train[:,7])
p=iknn.predict(test [:,0:6])
print(p)
print(confusion_matrix(test[:,7],p))
print("Accuracy_score is: ", accuracy_score(test[:,7],p))
score=list()
for i in range(1,25):
  iknn=KNeighborsClassifier(n_neighbors=i)
  iknn.fit(train [:,0:6], train[:,7])
  p=iknn.predict(test [:,0:6])
  a=accuracy_score(test[:,7],p)
  score.append(a)
  

plt.plot(range(1,25),score)
plt.xlabel("Value of k ")
plt.ylabel("Accuracy_score")
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd

train2=pd.read_csv("/home/ai7/My_Codes/ML/winequality-red.csv", delimiter=";")
test2=pd.read_csv("/home/ai7/My_Codes/ML/winequality-red.csv", delimiter=";")
print(train1)
print(test1)

X_train,X_test,y_train,y_test=train_test_split(train2,test2)

train=train2.as_matrix()
test=test2.as_matrix()
print(train.shape)
print(test.shape)

jknn=KNeighborsClassifier(n_neighbors=1)
jknn.fit(train [:,0:6], train[:,7])
p=jknn.predict(test [:,0:6])
print(p)
print(confusion_matrix(test[:,7],p))
print("Accuracy_score is: ", accuracy_score(test[:,7],p))


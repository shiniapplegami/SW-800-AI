from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd

train2=pd.read_csv("winequality-red.csv", delimiter=";")
test2=pd.read_csv("winequality-red.csv", delimiter=";")
print(train2)
print(test2)

#Splitting the data

X_train,X_test,y_train,y_test=train_test_split(train2,test2)

train=train2.as_matrix()
test=test2.as_matrix()
print(train.shape)
print(test.shape)

#Using KNN

jknn=KNeighborsClassifier(n_neighbors=1)
jknn.fit(train2 [:,0:6], train2[:,7])
p=jknn.predict(test2 [:,0:6])
print(p)
print(confusion_matrix(test[:,7],p))
print("Accuracy_score is: ", accuracy_score(test[:,7],p))

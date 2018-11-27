from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

iris=load_iris()
print(iris.data)
print(iris.target)

#Splitting the data

X_train,X_test,y_train,y_test=train_test_split(X,y)
print(X_train.shape)
print(X_test.shape)

#Using KNN

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

p=knn.predict(X_test)

#Plot

print(confusion_matrix(y_test,p))
print(accuracy_score(y_test,p))

print(p)

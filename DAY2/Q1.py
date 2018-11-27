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







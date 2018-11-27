from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,cross_val_score

iris=load_iris()

p=iris.data
q=iris.target

#Splitting the data

X_train,X_test,y_train,y_test=train_test_split(p,q)


#Using KNN

knn=KNeighborsClassifier(n_neighbors=1)
score=cross_val_score(knn,p,q,cv=4)
print(score)

knn.fit(p,q)
t=knn.predict(q)



print(confusion_matrix(q,t))
print(accuracy_matrix(q,t))
print(classification_report(q,t))

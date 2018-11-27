from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

#Opening file

Data = pd.read_csv("mining.csv", delimiter = ",")

#Filling missing values

Data.fillna(F.mean(), inplace=True)

Q = Data.as_matrix()

#Using KNN

knn = KNeighborsClassifier(n_neighbors=4)
le = preprocessing.LabelEncoder()
le.fit(Q[:,1])
k = le.transform(Q[:,1])

Q[:,1] = k

X = Q[:,[1,3,4,5,6,7,8,9,10]]
y = Q[:, 0]

#Converting into int data type

y=y.astype('int')

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X,y)

y_pred = knn.predict([[12, 2807, 90.25, 0.346, 11.5, 20.23, 3.1, 1, 0.34]])

print(y_pred)

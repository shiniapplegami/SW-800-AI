from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import matplotlib.pyplot as plt
import numpy as np

data=load_digits()

X=data.images
y=data.target

labe=list(zip(X,y))

for index,(image,label) in enumerate (labe[:6]):
    plt.subplot(3,6,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title(label)
    

#print(image)
plt.show()
#print(y)

q=len(X)

X1=X.reshape(q,-1)            

X_train=X1[:q//2]
y_train=y[:q//2]
X_test=X1[q//2:]
y_test=y[q//2:]

model=SVC(gamma=0.001)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print(y_pred)

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

labe=list(zip(X//2,y_pred))

for index,(image,label) in enumerate (labe[:8]):
    plt.subplot(3,6,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title(label)

plt.show()


q=len(X)
                                  
            

X_train=X[:q//2]
y_train=y[:q//2]
X_test=X[q//2:]
y_test=y[q//2:]

model=SVC(gamma=0.001)
model.fit(X_train,y_train)

y_pred1=model.predict(X_test)
print(y_pred1)



#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

#data=pd.read_csv("Advertising.csv")
#print (data.head())
#data=data.as_matrix()

#X=data[:,[1,2,3]]
#y=data[:,-1]

#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#lr=LinearRegression()
#lr.fit(X_train, y_train)
#p=lr.predict(X_test) 

#print(p)
#plt.scatter(y_test,p)
#plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

#data=pd.read_csv("loan.csv")
#print (data.head())
#data=data.as_matrix()

#X=data[:,[1,2]]
#y=data[:,-2]

#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#lr1=LinearRegression()
#lr1.fit(X_train, y_train)
#p1=lr1.predict(X_test)

#print(p1)
#plt.scatter(y_test,p1)
#plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

#data=pd.read_csv("pimaindians.csv")
#print (data.head())
#data=data.as_matrix()

#X=data[:, 0:8]
#y=data[:,-1]

#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#lg=LogisticRegression()
#lg.fit(X_train, y_train)
#p1=lg.predict(X_test)

#print(p1)
#plt.scatter(y_test,p1)
#plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("Planets.txt", delimiter="\s")
print(data)
print (data.head())
data=data.as_matrix()

X=data[:, 1:2]
y=data[:,-2]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

lg=LinearRegression()
lg.fit(X_train, y_train)
p1=lg.predict(X_test)

print(p1)
plt.scatter(y_test,p1)
plt.show()

import sklearn.metrics
print(metrics.mean_absolute_error(y_test,p))
print(metrics.mean_squared_error(y_test,p))
print(np.sqrt(metrics.mean_absolute_error(y_test,p)))
'''
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("city.csv")
print(data)
print (data.head())
data=data.as_matrix()

X=data[:, 1:6]
y=data[:,-2]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

lg=LinearRegression()
lg.fit(X_train, y_train)
p1=lg.predict(X_test)

print(p1)
plt.scatter(y_test,p1)
plt.show()
'''

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,cross_val_score

iris=load_iris()

p=iris.data
q=iris.target

knn=KNeighborsClassifier(n_neighbors=1)
score=cross_val_score(knn,p,q,cv=4)
print(score)

knn.fit(p,q)
t=knn.predict(q)


X_train,X_test,y_train,y_test=train_test_split(p,q)


print(confusion_matrix(q,t))
print(accuracy_matrix(q,t))
print(classification_report(q,t))

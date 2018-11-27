from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
import pandas as pd
import numpy as np

#Reading the data

data=pd.read_csv('s1.txt',delimiter='\s+')
print(data)

d1=data['V1'].values
d2=data['V2'].values
X=np.array(zip(d1,d2))

#Simple KMeans

kmeans=KMeans(n_clusters=15)

kmeans.fit(X)

y_kmeans=kmeans.predict(X)

#plot the graph

plt.scatter(X[:,0],X[:,1],s=50)
plt.show()
#centers=kmeans.cluster_centers_


plt.scatter(X[:,0],X[:,1],s=50,c=y_kmeans,cmap='viridis')

plt.show()

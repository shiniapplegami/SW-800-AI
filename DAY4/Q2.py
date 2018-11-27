#from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_moons
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt


#X,y_true=make_blobs(n_samples=300, centers=4, cluster_std=0.6)

X,y_true=make_moons(200,noise=0.05)

#Simple KMeans

kmeans=KMeans(2)
kmeans.fit(X)
y_means=kmeans.predict(X)

plt.scatter(X[:,0],X[:,1],s=50,c=y_means)

plt.show()

#Spectral Clustering

model=SpectralClustering(2,affinity='nearest_neighbors')
#model=SpectralClustering(2,affinity='nearest_neighbors',assign_labels='kmeans')
labels=model.fit_predict(X)

#Plot graph

plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis')
plt.show()

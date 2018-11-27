
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


X,y_true=make_blobs(n_samples=300, centers=4, cluster_std=0.6)
kmeans=KMeans(n_clusters=4)
kmeans.fit(X)
y_means=kmeans.predict(X)

plt.scatter(X[:,0],X[:,1],s=50,c=y_means,cmap='viridis')

plt.show()







import matplotlib.pyplot as plt
from sklearn import metrics, datasets, cluster, mixture
from sklearn.decomposition import PCA

# Load the breast_cancer dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# K-means clustering with 2 clusters
kmeans_algo = cluster.KMeans(n_clusters=2, algorithm='elkan', n_init=10)
kmeans_model = kmeans_algo.fit(X)
kmeans_labels = kmeans_model.labels_
kmeans_silhouette = metrics.silhouette_score(X, kmeans_labels, metric='euclidean')

# EM clustering with 2 clusters
em_algo = mixture.GaussianMixture(n_components=2, covariance_type='full', n_init=10)
em_model = em_algo.fit(X)
em_labels = em_model.predict(X)
em_silhouette = metrics.silhouette_score(X, em_labels, metric='euclidean')

# Perform PCA with two components
pca = PCA(n_components=2)
X_pca = pca.fit(X).transform(X)

# Scatter plot of PCA mapped data
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.title("PCA Scatter Plot")
plt.show()

print("K-Means Silhouette Score: ", kmeans_silhouette)
print("EM Silhouette Score: ", em_silhouette)


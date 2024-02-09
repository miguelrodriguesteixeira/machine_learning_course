import matplotlib.pyplot as plt
from sklearn import metrics, datasets, cluster, mixture
from sklearn.decomposition import PCA

# Load the wine dataset
data = datasets.load_wine()
X, y = data.data, data.target

# Initialize a list to store silhouette scores for k-means and EM
kmeans_silhouettes = []
em_silhouettes = []

# Try different values of k (number of clusters)
k_values = [2, 3, 4]

for k in k_values:
    # K-means clustering
    kmeans_algo = cluster.KMeans(n_clusters=k, algorithm='elkan', n_init=10)
    kmeans_model = kmeans_algo.fit(X)
    kmeans_labels = kmeans_model.labels_
    kmeans_silhouette = metrics.silhouette_score(X, kmeans_labels, metric='euclidean')
    kmeans_silhouettes.append(kmeans_silhouette)

    # EM clustering
    em_algo = mixture.GaussianMixture(n_components=k, covariance_type='full', n_init=10)
    em_model = em_algo.fit(X)
    em_labels = em_model.predict(X)
    em_silhouette = metrics.silhouette_score(X, em_labels, metric='euclidean')
    em_silhouettes.append(em_silhouette)

# Display the silhouette scores for k-means and EM with different k values
for i, k in enumerate(k_values):
    print(f'K-Means with {k} clusters - Silhouette: {kmeans_silhouettes[i]}')
    print(f'EM with {k} clusters - Silhouette: {em_silhouettes[i]}')

# Now, perform PCA with two components and repeat the clustering experiments
pca = PCA(n_components=2)
X_pca = pca.fit(X).transform(X)

# Initialize new lists for silhouette scores with PCA
kmeans_silhouettes_pca = []
em_silhouettes_pca = []

for k in k_values:
    # K-means clustering with PCA
    kmeans_algo_pca = cluster.KMeans(n_clusters=k, algorithm='elkan', n_init=10)
    kmeans_model_pca = kmeans_algo_pca.fit(X_pca)
    kmeans_labels_pca = kmeans_model_pca.labels_
    kmeans_silhouette_pca = metrics.silhouette_score(X_pca, kmeans_labels_pca, metric='euclidean')
    kmeans_silhouettes_pca.append(kmeans_silhouette_pca)

    # EM clustering with PCA
    em_algo_pca = mixture.GaussianMixture(n_components=k, covariance_type='full', n_init=10)
    em_model_pca = em_algo_pca.fit(X_pca)
    em_labels_pca = em_model_pca.predict(X_pca)
    em_silhouette_pca = metrics.silhouette_score(X_pca, em_labels_pca, metric='euclidean')
    em_silhouettes_pca.append(em_silhouette_pca)

# Display the silhouette scores for k-means and EM with PCA and different k values
for i, k in enumerate(k_values):
    print(f'K-Means with PCA and {k} clusters - Silhouette: {kmeans_silhouettes_pca[i]}')
    print(f'EM with PCA and {k} clusters - Silhouette: {em_silhouettes_pca[i]}')


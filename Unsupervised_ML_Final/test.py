import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import ParameterGrid

# Load dataset
data = pd.read_csv('Data\processed_music_data.csv')

# Inspect the first few rows and summary statistics
print(data.head())
print(data.info())
print(data.describe())

# Separate features and labels
X = data.iloc[:, :-1]  # all columns except the last
y_true = data.iloc[:, -1]  # the music genre column


# Histograms for feature distributions
X.hist(bins=30, figsize=(12, 10))
plt.tight_layout()
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.show()

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.title('PCA Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Elbow Method
inertia = []
silhouette_scores = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)

# Plot the results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, marker='o')
plt.title('Silhouette Scores')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.title('Clusters in PCA Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

final_silhouette = silhouette_score(X, cluster_labels)
print("Final Silhouette Score:", final_silhouette)

# If true labels are available, evaluate with ARI
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(y_true, cluster_labels)
print("Adjusted Rand Index (ARI):", ari)

# Cluster centers (in scaled space)
print("Cluster Centers:\n", kmeans.cluster_centers_)

# If you want to see how clusters align with true genres:
cluster_vs_genre = pd.crosstab(y_true, cluster_labels, rownames=['True Genre'], colnames=['Cluster'])
print(cluster_vs_genre)

# Define a function to reduce dimensions, cluster using K-Means, and visualize/evaluate the results
def reduce_and_cluster(X_reduced, method_name):
    # Use K-Means clustering (adjust n_clusters if needed)
    k = 4  # example: using 4 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_reduced)
    
    # Evaluate the clustering
    sil_score = silhouette_score(X_reduced, cluster_labels)
    ari = adjusted_rand_score(y_true, cluster_labels)
    
    # Plot the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    plt.title(f'{method_name} - Clustering Results')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(label='Cluster')
    plt.show()
    
    # Print evaluation metrics
    print(f"{method_name} - Silhouette Score: {sil_score:.4f}")
    print(f"{method_name} - Adjusted Rand Index: {ari:.4f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
reduce_and_cluster(X_pca, "PCA")

# Define a grid of K-Means parameters to try
param_grid = {
    'n_init': [10, 20, 50],
    'max_iter': [300, 500, 1000],
    'tol': [1e-4, 1e-3, 1e-2],
}

best_score = -1
best_params = None

for params in ParameterGrid(param_grid):
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, **params)
    cluster_labels = kmeans.fit_predict(X_pca)
    sil = silhouette_score(X_pca, cluster_labels)
    if sil > best_score:
        best_score = sil
        best_params = params

print("Best K-Means parameters based on Silhouette Score:")
print(best_params)
print("Best Silhouette Score:", best_score)

pca_scores = {}
for whiten in [False, True]:
    pca_temp = PCA(n_components=2, whiten=whiten)
    X_pca_temp = pca_temp.fit_transform(X_pca)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca_temp)
    sil = silhouette_score(X_pca_temp, cluster_labels)
    pca_scores[whiten] = sil
    print(f"PCA with whiten={whiten} -> Silhouette Score: {sil:.4f}")

for init_method in ['k-means++', 'random']:
    kmeans = KMeans(n_clusters=optimal_k, init=init_method, random_state=42, n_init=best_params.get('n_init', 10))
    cluster_labels = kmeans.fit_predict(X_pca)
    sil = silhouette_score(X_pca, cluster_labels)
    ari = adjusted_rand_score(y_true, cluster_labels)
    print(f"Init method: {init_method} -> Silhouette Score: {sil:.4f}, ARI: {ari:.4f}")

# Unsupervised Learning - Complete Guide

## ⚡ Interview Quick Summary

> **Core insight**: Unsupervised learning finds structure in unlabeled data. Clustering groups similar points; dimensionality reduction finds compact representations. Both require careful evaluation since there's no ground truth.

### Clustering Algorithm Comparison

| Algorithm | Cluster Shape | Scales | Handles Noise | Needs k? |
|-----------|--------------|--------|---------------|----------|
| K-Means | Spherical, equal size | O(nkd) | No | Yes |
| DBSCAN | Arbitrary shape | O(n log n) | Yes (noise points) | No |
| Hierarchical | Any (dendrogram) | O(n² log n) | Partial | No (cut level) |
| GMM | Elliptical (soft assign) | O(nkd) | No | Yes |
| HDBSCAN | Arbitrary, variable density | O(n log n) | Yes | No |

### K-Means — Must Know Internals

```
Algorithm:
  1. Initialize k centroids (random++ for better init: k-means++)
  2. Assign each point to nearest centroid (E-step)
  3. Update centroids = mean of assigned points (M-step)
  4. Repeat until convergence

Key properties:
  - Minimizes within-cluster sum of squares (WCSS)
  - Guaranteed to converge but may find local minimum (run multiple times!)
  - Sensitive to initialization → use k-means++ or run 10x with different seeds
  - Fails with non-spherical clusters, very different densities

Choosing k:
  Elbow method: plot WCSS vs k, choose the "elbow"
  Silhouette score: measures how similar point is to own cluster vs neighbors
  Silhouette ∈ [-1, 1], higher = better (>0.5 is good)
```

### Dimensionality Reduction Comparison

```
PCA:     Linear, fast, global structure, interpretable axes
         Use: preprocessing for ML, multicollinearity removal

t-SNE:   Non-linear, for visualization only, perplexity controls local structure
         CANNOT add new points, non-deterministic, don't interpret distances

UMAP:    Non-linear, faster than t-SNE, preserves more global structure
         Better for clustering visualization, can add new points

Autoencoder: Learns compressed representation via reconstruction
             Flexible, task-specific, good for images
```

### 🚨 Top Interview Pitfalls
- Not normalizing features before K-Means (distance-based, sensitive to scale)
- Saying t-SNE distances are meaningful — they're **NOT**, only cluster membership can be interpreted
- Forgetting that K-Means needs k specified — DBSCAN is better when you don't know k
- Not mentioning **silhouette score** for evaluating clustering (no labels needed!)
- Confusing **PCA for visualization** (OK) vs **PCA for feature engineering** before supervised learning (be careful — may discard discriminative features)

---

## Table of Contents
1. [Clustering Algorithms](#clustering-algorithms)
2. [Dimensionality Reduction](#dimensionality-reduction)
3. [Anomaly Detection](#anomaly-detection)
4. [Association Rules](#association-rules)
5. [Interview Questions](#interview-questions)

---

## 1. Clustering Algorithms

### K-Means Clustering

```python
import numpy as np
from sklearn.cluster import KMeans

class KMeansScratch:
    """
    Partition n observations into k clusters.
    Each observation belongs to cluster with nearest centroid.
    
    Algorithm:
    1. Initialize k centroids randomly
    2. Assign each point to nearest centroid
    3. Update centroids as mean of assigned points
    4. Repeat until convergence
    
    Objective: Minimize within-cluster sum of squares (WCSS/Inertia)
    """
    
    def __init__(self, n_clusters=3, max_iters=300, tol=1e-4, 
                 init='k-means++', n_init=10):
        self.k = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.n_init = n_init
    
    def _init_centroids(self, X):
        if self.init == 'random':
            idx = np.random.choice(len(X), self.k, replace=False)
            return X[idx].copy()
        elif self.init == 'k-means++':
            # K-means++ initialization for better convergence
            centroids = [X[np.random.randint(len(X))]]
            
            for _ in range(1, self.k):
                # Distance to nearest centroid for each point
                distances = np.min([
                    np.sum((X - c) ** 2, axis=1) for c in centroids
                ], axis=0)
                
                # Select next centroid with probability proportional to distance²
                probs = distances / distances.sum()
                next_idx = np.random.choice(len(X), p=probs)
                centroids.append(X[next_idx])
            
            return np.array(centroids)
    
    def _assign_clusters(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        new_centroids = np.array([
            X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 
            else self.centroids[k]
            for k in range(self.k)
        ])
        return new_centroids
    
    def fit(self, X):
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        
        # Run multiple times with different initializations
        for _ in range(self.n_init):
            self.centroids = self._init_centroids(X)
            
            for _ in range(self.max_iters):
                labels = self._assign_clusters(X)
                new_centroids = self._update_centroids(X, labels)
                
                # Check convergence
                if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                    break
                self.centroids = new_centroids
            
            # Calculate inertia
            inertia = sum(
                np.sum((X[labels == k] - self.centroids[k]) ** 2)
                for k in range(self.k)
            )
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = self.centroids.copy()
                best_labels = labels.copy()
        
        self.centroids = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self
    
    def predict(self, X):
        return self._assign_clusters(X)


# Choosing optimal k

# Method 1: Elbow Method
def elbow_method(X, k_range=range(1, 11)):
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    # Plot and look for "elbow" point
    return inertias

# Method 2: Silhouette Score
from sklearn.metrics import silhouette_score, silhouette_samples

def silhouette_analysis(X, k_range=range(2, 11)):
    """
    Silhouette score: measures how similar point is to its cluster vs other clusters.
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    - a(i): mean distance to other points in same cluster
    - b(i): mean distance to points in nearest other cluster
    - Range: [-1, 1], higher is better
    """
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
    return scores

# Method 3: Gap Statistic
def gap_statistic(X, k_range=range(1, 11), n_refs=10):
    """
    Compare within-cluster dispersion to expected under null reference.
    """
    gaps = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        
        # Reference datasets (uniform random)
        ref_dispersions = []
        for _ in range(n_refs):
            random_ref = np.random.uniform(
                X.min(axis=0), X.max(axis=0), X.shape
            )
            kmeans_ref = KMeans(n_clusters=k, random_state=42)
            kmeans_ref.fit(random_ref)
            ref_dispersions.append(np.log(kmeans_ref.inertia_))
        
        gap = np.mean(ref_dispersions) - np.log(kmeans.inertia_)
        gaps.append(gap)
    
    return gaps
```

### DBSCAN (Density-Based Clustering)

```python
import numpy as np
from sklearn.cluster import DBSCAN
from collections import deque

class DBSCANScratch:
    """
    Density-Based Spatial Clustering of Applications with Noise.
    
    Concepts:
    - Core point: has at least min_samples within eps
    - Border point: within eps of a core point, but not core itself
    - Noise point: neither core nor border
    
    Advantages:
    - No need to specify number of clusters
    - Can find arbitrarily shaped clusters
    - Robust to outliers (labels them as noise)
    
    Parameters:
    - eps: maximum distance between points to be neighbors
    - min_samples: minimum points to form a dense region
    """
    
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
    
    def _get_neighbors(self, X, point_idx):
        distances = np.sqrt(np.sum((X - X[point_idx]) ** 2, axis=1))
        return np.where(distances <= self.eps)[0]
    
    def fit_predict(self, X):
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1)  # -1 = unvisited/noise
        cluster_id = 0
        
        for i in range(n_samples):
            if labels[i] != -1:  # Already processed
                continue
            
            neighbors = self._get_neighbors(X, i)
            
            if len(neighbors) < self.min_samples:
                # Noise point (for now, might become border later)
                continue
            
            # Start new cluster
            labels[i] = cluster_id
            
            # Expand cluster using BFS
            seed_set = deque(neighbors)
            seed_set.remove(i)
            
            while seed_set:
                j = seed_set.popleft()
                
                if labels[j] == -1:  # Was noise, now border
                    labels[j] = cluster_id
                
                if labels[j] != -1:  # Already in a cluster
                    continue
                
                labels[j] = cluster_id
                j_neighbors = self._get_neighbors(X, j)
                
                if len(j_neighbors) >= self.min_samples:
                    # j is a core point, add its neighbors
                    for k in j_neighbors:
                        if labels[k] == -1:
                            seed_set.append(k)
            
            cluster_id += 1
        
        self.labels_ = labels
        return labels


# HDBSCAN (Hierarchical DBSCAN)
"""
Improvements over DBSCAN:
- Doesn't require eps parameter
- Finds clusters of varying densities
- Extracts flat clustering from hierarchy

pip install hdbscan
"""
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    cluster_selection_epsilon=0.0,
    metric='euclidean',
    cluster_selection_method='eom'  # 'eom' or 'leaf'
)
labels = clusterer.fit_predict(X)
```

### Hierarchical Clustering

```python
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

class AgglomerativeClusteringScratch:
    """
    Bottom-up hierarchical clustering.
    
    Algorithm:
    1. Start with each point as its own cluster
    2. Find two closest clusters
    3. Merge them
    4. Repeat until desired number of clusters
    
    Linkage methods:
    - Single: min distance between clusters (chaining problem)
    - Complete: max distance between clusters (compact clusters)
    - Average: average distance between all pairs
    - Ward: minimizes total within-cluster variance
    """
    
    def __init__(self, n_clusters=2, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage_type = linkage
    
    def _distance_between_clusters(self, X, cluster1, cluster2):
        points1 = X[cluster1]
        points2 = X[cluster2]
        
        if self.linkage_type == 'single':
            # Minimum distance
            return np.min([
                np.sqrt(np.sum((p1 - p2) ** 2))
                for p1 in points1 for p2 in points2
            ])
        elif self.linkage_type == 'complete':
            # Maximum distance
            return np.max([
                np.sqrt(np.sum((p1 - p2) ** 2))
                for p1 in points1 for p2 in points2
            ])
        elif self.linkage_type == 'average':
            # Average distance
            distances = [
                np.sqrt(np.sum((p1 - p2) ** 2))
                for p1 in points1 for p2 in points2
            ]
            return np.mean(distances)
        elif self.linkage_type == 'ward':
            # Increase in variance if merged
            merged = np.vstack([points1, points2])
            centroid = merged.mean(axis=0)
            return np.sum((merged - centroid) ** 2)
    
    def fit_predict(self, X):
        n_samples = X.shape[0]
        # Start with each point as its own cluster
        clusters = [[i] for i in range(n_samples)]
        
        while len(clusters) > self.n_clusters:
            # Find closest pair of clusters
            min_dist = np.inf
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._distance_between_clusters(X, clusters[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            # Merge clusters
            clusters[merge_i].extend(clusters[merge_j])
            del clusters[merge_j]
        
        # Create labels
        labels = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for point_idx in cluster:
                labels[point_idx] = cluster_id
        
        return labels


# Using scipy for dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Compute linkage matrix
Z = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

# Cut dendrogram to get clusters
labels = fcluster(Z, t=3, criterion='maxclust')  # t = number of clusters
# or
labels = fcluster(Z, t=5.0, criterion='distance')  # t = distance threshold
```

### Gaussian Mixture Models (GMM)

```python
import numpy as np
from sklearn.mixture import GaussianMixture

class GMMScratch:
    """
    Soft clustering using mixture of Gaussians.
    Each point has probability of belonging to each cluster.
    
    Model: P(x) = Σₖ πₖ N(x | μₖ, Σₖ)
    - πₖ: mixing coefficient (prior probability of cluster k)
    - μₖ: mean of cluster k
    - Σₖ: covariance of cluster k
    
    Training: Expectation-Maximization (EM) algorithm
    """
    
    def __init__(self, n_components=3, max_iters=100, tol=1e-4):
        self.k = n_components
        self.max_iters = max_iters
        self.tol = tol
    
    def _gaussian_pdf(self, X, mean, cov):
        n_features = X.shape[1]
        diff = X - mean
        
        # Handle numerical issues
        cov = cov + 1e-6 * np.eye(n_features)
        
        coef = 1 / np.sqrt((2 * np.pi) ** n_features * np.linalg.det(cov))
        exponent = -0.5 * np.sum(diff @ np.linalg.inv(cov) * diff, axis=1)
        
        return coef * np.exp(exponent)
    
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.ones(self.k) / self.k  # πₖ
        random_idx = np.random.choice(n_samples, self.k, replace=False)
        self.means = X[random_idx]  # μₖ
        self.covariances = [np.eye(n_features) for _ in range(self.k)]  # Σₖ
        
        log_likelihood_old = 0
        
        for iteration in range(self.max_iters):
            # E-step: compute responsibilities (posterior probabilities)
            responsibilities = np.zeros((n_samples, self.k))
            
            for k in range(self.k):
                responsibilities[:, k] = (
                    self.weights[k] * 
                    self._gaussian_pdf(X, self.means[k], self.covariances[k])
                )
            
            # Normalize
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)
            
            # M-step: update parameters
            Nk = responsibilities.sum(axis=0)  # Effective number of points per cluster
            
            for k in range(self.k):
                # Update mean
                self.means[k] = (responsibilities[:, k] @ X) / Nk[k]
                
                # Update covariance
                diff = X - self.means[k]
                self.covariances[k] = (
                    (responsibilities[:, k][:, np.newaxis] * diff).T @ diff / Nk[k]
                )
                
                # Update weight
                self.weights[k] = Nk[k] / n_samples
            
            # Check convergence (log likelihood)
            log_likelihood = np.sum(np.log(responsibilities.sum(axis=1) + 1e-10))
            
            if abs(log_likelihood - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood
        
        self.responsibilities_ = responsibilities
        return self
    
    def predict(self, X):
        responsibilities = np.zeros((X.shape[0], self.k))
        
        for k in range(self.k):
            responsibilities[:, k] = (
                self.weights[k] * 
                self._gaussian_pdf(X, self.means[k], self.covariances[k])
            )
        
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X):
        responsibilities = np.zeros((X.shape[0], self.k))
        
        for k in range(self.k):
            responsibilities[:, k] = (
                self.weights[k] * 
                self._gaussian_pdf(X, self.means[k], self.covariances[k])
            )
        
        return responsibilities / responsibilities.sum(axis=1, keepdims=True)


# Sklearn GMM
gmm = GaussianMixture(
    n_components=3,
    covariance_type='full',  # 'full', 'tied', 'diag', 'spherical'
    max_iter=100,
    n_init=1,
    init_params='kmeans'
)
gmm.fit(X)

# Model selection with BIC/AIC
bic_scores = []
aic_scores = []
for n in range(1, 11):
    gmm = GaussianMixture(n_components=n)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    aic_scores.append(gmm.aic(X))

optimal_n = np.argmin(bic_scores) + 1
```

---

## 2. Dimensionality Reduction

### Principal Component Analysis (PCA)

```python
import numpy as np
from sklearn.decomposition import PCA

class PCAScratch:
    """
    Linear dimensionality reduction using orthogonal transformation.
    Projects data onto directions of maximum variance.
    
    Algorithm:
    1. Center the data (subtract mean)
    2. Compute covariance matrix
    3. Find eigenvalues and eigenvectors
    4. Sort by eigenvalue (descending)
    5. Project onto top k eigenvectors
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
    
    def fit(self, X):
        n_samples, n_features = X.shape
        
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)
        
        # Step 1: Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Step 2: Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Step 3: Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Step 4: Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Step 5: Select top k components
        self.components_ = eigenvectors[:, :self.n_components].T  # (k, n_features)
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = (
            self.explained_variance_ / np.sum(eigenvalues)
        )
        
        return self
    
    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        return X_transformed @ self.components_ + self.mean_


# Using SVD for PCA (more numerically stable)
class PCA_SVD:
    def __init__(self, n_components=None):
        self.n_components = n_components
    
    def fit_transform(self, X):
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        
        # SVD: X = U @ S @ Vt
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        if self.n_components is None:
            self.n_components = min(X.shape)
        
        self.components_ = Vt[:self.n_components]
        self.explained_variance_ = (S[:self.n_components] ** 2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / self.explained_variance_.sum()
        )
        
        return X_centered @ self.components_.T


# Choosing number of components
pca = PCA()
pca.fit(X)

# Method 1: Cumulative explained variance
cumsum = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumsum >= 0.95) + 1

# Method 2: Elbow in explained variance
# Plot explained_variance_ratio_ and look for elbow

# Incremental PCA (for large datasets)
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=50, batch_size=200)
for batch in batches:
    ipca.partial_fit(batch)

# Kernel PCA (non-linear)
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_kpca = kpca.fit_transform(X)
```

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

```python
import numpy as np
from sklearn.manifold import TSNE

"""
t-SNE: Non-linear dimensionality reduction for visualization.

Algorithm:
1. Compute pairwise similarities in high-dim (Gaussian kernel)
2. Define similarities in low-dim (t-distribution)
3. Minimize KL divergence between the two distributions

Key properties:
- Preserves local structure well
- Non-deterministic (different runs give different results)
- Perplexity controls balance between local and global structure
- Computationally expensive O(n²)

Limitations:
- Mainly for visualization (2-3 dims)
- Can't transform new points (no transform method)
- Distances in output don't have meaning
- Cluster sizes don't reflect true sizes
"""

tsne = TSNE(
    n_components=2,
    perplexity=30,         # Balance between local/global (5-50)
    early_exaggeration=12, # Spacing between clusters
    learning_rate='auto',  # or 200
    n_iter=1000,
    metric='euclidean',
    init='pca',           # or 'random'
    random_state=42,
    n_jobs=-1
)

X_tsne = tsne.fit_transform(X)

# Tips for using t-SNE:
# 1. Try different perplexity values
# 2. Run multiple times to check stability
# 3. Don't over-interpret cluster sizes
# 4. Use PCA first to reduce to ~50 dims for speed
```

### UMAP (Uniform Manifold Approximation and Projection)

```python
"""
UMAP: Modern alternative to t-SNE.

Advantages over t-SNE:
- Preserves more global structure
- Much faster
- Can transform new data points
- Better scalability

pip install umap-learn
"""

import umap

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,        # Local vs global (similar to perplexity)
    min_dist=0.1,          # Minimum distance between points in output
    metric='euclidean',
    spread=1.0,
    random_state=42
)

X_umap = reducer.fit_transform(X)

# Transform new data
X_new_umap = reducer.transform(X_new)

# Supervised UMAP
reducer_supervised = umap.UMAP()
X_umap_supervised = reducer_supervised.fit_transform(X, y=labels)
```

### Other Dimensionality Reduction Methods

```python
from sklearn.decomposition import TruncatedSVD, NMF, FactorAnalysis
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding

# Truncated SVD (for sparse matrices, like TF-IDF)
svd = TruncatedSVD(n_components=100)
X_svd = svd.fit_transform(X_sparse)

# Non-negative Matrix Factorization (for non-negative data)
nmf = NMF(n_components=10, init='nndsvd')
W = nmf.fit_transform(X)  # Document-topic matrix
H = nmf.components_        # Topic-word matrix

# Factor Analysis (similar to PCA but models noise)
fa = FactorAnalysis(n_components=10)
X_fa = fa.fit_transform(X)

# Multidimensional Scaling (preserve pairwise distances)
mds = MDS(n_components=2, dissimilarity='euclidean')
X_mds = mds.fit_transform(X)

# Isomap (non-linear, geodesic distances)
isomap = Isomap(n_neighbors=10, n_components=2)
X_isomap = isomap.fit_transform(X)

# Locally Linear Embedding
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2)
X_lle = lle.fit_transform(X)
```

---

## 3. Anomaly Detection

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

# Isolation Forest
"""
Isolates anomalies by random recursive partitioning.
Anomalies are easier to isolate (fewer splits needed).

Key insight: Anomalies have shorter average path length in trees.
"""

iso_forest = IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination=0.1,     # Expected proportion of outliers
    random_state=42
)
predictions = iso_forest.fit_predict(X)  # 1 = normal, -1 = anomaly
scores = iso_forest.score_samples(X)      # Lower = more anomalous


# Local Outlier Factor (LOF)
"""
Compares local density of a point to its neighbors.
LOF > 1 indicates lower density than neighbors (outlier).
"""

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.1,
    novelty=False  # True for novelty detection (fit then predict)
)
predictions = lof.fit_predict(X)
scores = lof.negative_outlier_factor_


# One-Class SVM
"""
Learns a boundary around normal data.
New points outside boundary are anomalies.
"""

ocsvm = OneClassSVM(
    kernel='rbf',
    gamma='scale',
    nu=0.1  # Upper bound on fraction of outliers
)
ocsvm.fit(X_normal)
predictions = ocsvm.predict(X_test)


# Elliptic Envelope (Gaussian assumption)
"""
Assumes data is Gaussian distributed.
Fits robust covariance matrix and detects outliers.
"""

ee = EllipticEnvelope(contamination=0.1)
predictions = ee.fit_predict(X)


# Statistical methods
def zscore_anomalies(X, threshold=3):
    """Z-score based anomaly detection."""
    z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
    return np.any(z_scores > threshold, axis=1)

def iqr_anomalies(X, multiplier=1.5):
    """IQR based anomaly detection."""
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return np.any((X < lower) | (X > upper), axis=1)


# Autoencoders for anomaly detection
"""
Train autoencoder on normal data.
High reconstruction error indicates anomaly.
"""

import tensorflow as tf

def autoencoder_anomaly_detector(X_train, X_test, encoding_dim=32):
    input_dim = X_train.shape[1]
    
    # Encoder
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(encoding_dim, activation='relu')
    ])
    
    # Decoder
    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(input_dim, activation='linear')
    ])
    
    # Autoencoder
    autoencoder = tf.keras.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, X_train, epochs=50, verbose=0)
    
    # Reconstruction error
    reconstructions = autoencoder.predict(X_test)
    errors = np.mean((X_test - reconstructions) ** 2, axis=1)
    
    # Threshold based on training reconstruction error
    train_recon = autoencoder.predict(X_train)
    train_errors = np.mean((X_train - train_recon) ** 2, axis=1)
    threshold = np.percentile(train_errors, 95)
    
    return errors > threshold
```

---

## 4. Association Rules

```python
"""
Association Rule Mining: Find patterns in transactional data.

Metrics:
- Support: frequency of itemset
- Confidence: P(B|A) = support(A,B) / support(A)
- Lift: P(A,B) / (P(A) * P(B)) = confidence(A→B) / support(B)
  - Lift > 1: positive correlation
  - Lift = 1: independent
  - Lift < 1: negative correlation

pip install mlxtend
"""

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# Sample transactions
transactions = [
    ['milk', 'bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'diapers', 'beer'],
    ['bread', 'butter'],
    ['milk', 'diapers', 'beer', 'bread'],
]

# Encode transactions
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

# Find frequent itemsets using Apriori
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Filter rules
high_lift_rules = rules[rules['lift'] > 1.0]
high_confidence_rules = rules[rules['confidence'] > 0.7]

# FP-Growth (faster alternative to Apriori)
from mlxtend.frequent_patterns import fpgrowth

frequent_itemsets_fp = fpgrowth(df, min_support=0.4, use_colnames=True)
```

---

## 5. Interview Questions

```python
# Q1: K-Means vs DBSCAN vs Hierarchical Clustering?
"""
K-Means:
- Pros: Fast, scalable, works well with spherical clusters
- Cons: Need to specify k, sensitive to initialization, only spherical

DBSCAN:
- Pros: No need for k, handles arbitrary shapes, identifies outliers
- Cons: Sensitive to eps and min_samples, struggles with varying densities

Hierarchical:
- Pros: Dendrogram visualization, no need for k upfront
- Cons: Slow O(n³), memory intensive, sensitive to noise
"""


# Q2: How to determine optimal number of clusters?
"""
1. Elbow Method: Plot inertia vs k, look for elbow
2. Silhouette Score: Higher score (closer to 1) is better
3. Gap Statistic: Compare within-cluster dispersion to reference
4. Domain knowledge: Sometimes k is determined by business needs
5. Stability: Run multiple times, check consistency
"""


# Q3: PCA vs t-SNE vs UMAP?
"""
PCA:
- Linear transformation
- Preserves global structure
- Interpretable components
- Can project new data

t-SNE:
- Non-linear
- Preserves local structure
- For visualization only
- Cannot project new data

UMAP:
- Non-linear
- Preserves both local and global
- Faster than t-SNE
- Can project new data
"""


# Q4: When to use GMM over K-Means?
"""
Use GMM when:
- Need soft assignments (probability of belonging to each cluster)
- Clusters are elliptical, not spherical
- Clusters have different sizes/densities
- Want to model uncertainty in cluster assignments

K-Means is special case of GMM with:
- Spherical, equal-size covariances
- Hard assignments
"""


# Q5: What is the curse of dimensionality in clustering?
"""
In high dimensions:
1. Data becomes sparse
2. Distance metrics become less meaningful (all points equidistant)
3. Clusters harder to separate
4. Overfitting risk increases

Solutions:
- Dimensionality reduction before clustering
- Use appropriate distance metrics
- Feature selection
"""


# Q6: Explain Isolation Forest algorithm
"""
1. Build random trees by:
   - Randomly select feature
   - Randomly select split value between min and max
   - Recurse until each point is isolated

2. Anomalies have shorter path lengths because:
   - They're in sparse regions
   - Few splits needed to isolate them

3. Average path length across trees → anomaly score
   - Shorter path = more anomalous
"""


# Q7: What is the difference between novelty detection and outlier detection?
"""
Outlier Detection:
- Training data contains outliers
- Detect outliers in training data
- LocalOutlierFactor with novelty=False

Novelty Detection:
- Training data is "clean" (no outliers)
- Detect if new points are different from training
- LocalOutlierFactor with novelty=True
- One-Class SVM
"""


# Q8: How does silhouette score work?
"""
For each point i:
- a(i) = mean distance to other points in same cluster
- b(i) = mean distance to points in nearest other cluster
- s(i) = (b(i) - a(i)) / max(a(i), b(i))

Interpretation:
- s(i) close to 1: well-clustered
- s(i) close to 0: on cluster boundary
- s(i) close to -1: probably in wrong cluster

Overall score: mean of all s(i)
"""


# Q9: What is the EM algorithm in GMM?
"""
E-step (Expectation):
- Compute responsibilities (posterior probabilities)
- γₙₖ = P(zₙ = k | xₙ) using current parameters

M-step (Maximization):
- Update parameters using responsibilities
- Update means: weighted average of points
- Update covariances: weighted covariance
- Update weights: proportion of responsibilities

Converges to local maximum of likelihood.
"""


# Q10: How to evaluate clustering when no ground truth?
"""
Internal metrics (no labels):
- Silhouette score: cluster cohesion vs separation
- Davies-Bouldin index: lower is better
- Calinski-Harabasz index: ratio of between/within cluster dispersion

External metrics (if labels available):
- Adjusted Rand Index
- Normalized Mutual Information
- V-measure, homogeneity, completeness

Visual inspection:
- Plot clusters (if 2D/3D or after reduction)
- Examine cluster characteristics
"""
```

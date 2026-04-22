# Machine Learning Fundamentals - Complete Guide

## 1. Mathematical Foundations

### Linear Algebra

#### Vectors & Matrices
```python
import numpy as np

# Vector operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

dot_product = np.dot(v1, v2)          # 32
cross_product = np.cross(v1, v2)      # [-3, 6, -3]
norm = np.linalg.norm(v1)             # L2 norm: sqrt(14)

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

matmul = A @ B                        # Matrix multiplication
transpose = A.T                       # Transpose
inverse = np.linalg.inv(A)            # Inverse
determinant = np.linalg.det(A)        # Determinant
```

#### Eigenvalues & Eigenvectors
```python
# Av = λv (eigenvector v, eigenvalue λ)
eigenvalues, eigenvectors = np.linalg.eig(A)

# Applications:
# - PCA (Principal Component Analysis)
# - PageRank algorithm
# - Stability analysis
```

#### SVD (Singular Value Decomposition)
```python
# A = U × Σ × V^T
U, S, Vt = np.linalg.svd(A)

# Applications:
# - Dimensionality reduction
# - Image compression
# - Recommendation systems (matrix factorization)
# - Latent Semantic Analysis
```

### Calculus

#### Gradients
```python
# Gradient: vector of partial derivatives
# ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

# Example: f(x, y) = x² + y²
# ∇f = [2x, 2y]

# In neural networks:
# - Gradients tell us direction of steepest ascent
# - We move in opposite direction (gradient descent)
```

#### Chain Rule
```python
# (f ∘ g)'(x) = f'(g(x)) · g'(x)

# In neural networks (backpropagation):
# ∂L/∂w = ∂L/∂y · ∂y/∂z · ∂z/∂w

# Example: y = relu(wx + b)
# ∂L/∂w = ∂L/∂y · relu'(z) · x
```

### Probability & Statistics

#### Probability Distributions
```python
import scipy.stats as stats

# Normal/Gaussian Distribution
normal = stats.norm(loc=0, scale=1)  # mean=0, std=1
normal.pdf(0)     # Probability density
normal.cdf(1.96)  # Cumulative distribution
normal.rvs(100)   # Random samples

# Bernoulli Distribution (binary outcomes)
bernoulli = stats.bernoulli(p=0.5)

# Binomial Distribution (n Bernoulli trials)
binomial = stats.binom(n=10, p=0.5)

# Poisson Distribution (rare events)
poisson = stats.poisson(mu=5)

# Exponential Distribution (time between events)
exponential = stats.expon(scale=1/5)
```

#### Bayes' Theorem
```python
# P(A|B) = P(B|A) × P(A) / P(B)

# Components:
# - P(A|B): Posterior probability
# - P(B|A): Likelihood
# - P(A): Prior probability
# - P(B): Evidence/Marginal likelihood

# Example: Spam Classification
# P(spam|words) = P(words|spam) × P(spam) / P(words)
```

#### Key Statistical Concepts
```python
# Central Limit Theorem:
# Sample means approach normal distribution as n → ∞

# Maximum Likelihood Estimation (MLE):
# Find parameters that maximize P(data|parameters)

# Maximum A Posteriori (MAP):
# Find parameters that maximize P(parameters|data)
# Includes prior: P(θ|data) ∝ P(data|θ) × P(θ)

# Confidence Intervals:
from scipy import stats
data = np.random.normal(100, 15, 100)
ci = stats.t.interval(0.95, len(data)-1, loc=np.mean(data), 
                       scale=stats.sem(data))

# Hypothesis Testing:
# H0: Null hypothesis (no effect)
# H1: Alternative hypothesis
# p-value: P(observing data | H0 is true)
# If p-value < α (typically 0.05), reject H0
```

---

## 2. Supervised Learning

### Linear Regression

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# From Scratch
class LinearRegressionScratch:
    def fit(self, X, y):
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Normal equation: θ = (X^T X)^(-1) X^T y
        self.theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        return self
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta

# Gradient Descent Implementation
class LinearRegressionGD:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            y_pred = X @ self.weights + self.bias
            
            # Gradients
            dw = (1/n_samples) * X.T @ (y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self
    
    def predict(self, X):
        return X @ self.weights + self.bias

# Regularization
# Ridge (L2): ||y - Xw||² + α||w||²
ridge = Ridge(alpha=1.0)

# Lasso (L1): ||y - Xw||² + α||w||₁
lasso = Lasso(alpha=1.0)

# ElasticNet: ||y - Xw||² + α₁||w||₁ + α₂||w||²
from sklearn.linear_model import ElasticNet
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
```

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)
            
            # Gradients (cross-entropy loss)
            dw = (1/n_samples) * X.T @ (y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self
    
    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# Loss function: Binary Cross-Entropy
# L = -1/n Σ [y log(p) + (1-y) log(1-p)]
```

### Decision Trees

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class DecisionTreeScratch:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    def gini(self, y):
        """Gini impurity: 1 - Σ p_i²"""
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def entropy(self, y):
        """Entropy: -Σ p_i log(p_i)"""
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def information_gain(self, parent, left, right):
        """IG = entropy(parent) - weighted_avg(entropy(children))"""
        n = len(parent)
        n_left, n_right = len(left), len(right)
        return (self.entropy(parent) - 
                (n_left/n * self.entropy(left) + 
                 n_right/n * self.entropy(right)))
    
    def best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if sum(left_mask) < self.min_samples_split:
                    continue
                if sum(right_mask) < self.min_samples_split:
                    continue
                
                gain = self.information_gain(
                    y, y[left_mask], y[right_mask]
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold

# Key hyperparameters:
# - max_depth: Maximum tree depth
# - min_samples_split: Minimum samples to split
# - min_samples_leaf: Minimum samples in leaf
# - max_features: Features to consider for split
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Key concepts:
# 1. Bagging: Bootstrap aggregating
# 2. Random feature selection at each split
# 3. Averaging predictions (regression) or voting (classification)

rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Max depth per tree
    max_features='sqrt',   # Features per split
    min_samples_split=2,
    bootstrap=True,        # Bootstrap sampling
    oob_score=True,        # Out-of-bag score
    n_jobs=-1              # Parallel processing
)

# Feature importance
importances = rf.feature_importances_
```

### Support Vector Machines

```python
from sklearn.svm import SVC, SVR

# Key concepts:
# - Find hyperplane that maximizes margin
# - Support vectors: points closest to decision boundary
# - Kernel trick: map to higher dimensions

svm = SVC(
    C=1.0,              # Regularization (lower = more regularization)
    kernel='rbf',       # 'linear', 'poly', 'rbf', 'sigmoid'
    gamma='scale',      # Kernel coefficient
    probability=True    # Enable probability estimates
)

# Kernels:
# Linear: K(x, y) = x · y
# Polynomial: K(x, y) = (γx·y + r)^d
# RBF: K(x, y) = exp(-γ||x-y||²)
# Sigmoid: K(x, y) = tanh(γx·y + r)
```

### K-Nearest Neighbors

```python
from sklearn.neighbors import KNeighborsClassifier

class KNNScratch:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        predictions = []
        for x in X:
            # Calculate distances
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            # Get k nearest indices
            k_indices = np.argsort(distances)[:self.k]
            # Get k nearest labels
            k_labels = self.y_train[k_indices]
            # Majority vote
            most_common = np.bincount(k_labels).argmax()
            predictions.append(most_common)
        return np.array(predictions)

# Distance metrics:
# Euclidean: sqrt(Σ(x_i - y_i)²)
# Manhattan: Σ|x_i - y_i|
# Minkowski: (Σ|x_i - y_i|^p)^(1/p)
```

### Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Naive Bayes assumption:
# Features are conditionally independent given the class
# P(x1, x2, ..., xn | y) = P(x1|y) × P(x2|y) × ... × P(xn|y)

class GaussianNBScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.parameters[c] = {
                'mean': X_c.mean(axis=0),
                'var': X_c.var(axis=0),
                'prior': len(X_c) / len(X)
            }
        return self
    
    def _gaussian_pdf(self, x, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * \
               np.exp(-(x - mean)**2 / (2 * var))
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.parameters[c]['prior'])
                likelihood = np.sum(np.log(
                    self._gaussian_pdf(
                        x,
                        self.parameters[c]['mean'],
                        self.parameters[c]['var']
                    ) + 1e-10
                ))
                posteriors.append(prior + likelihood)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)

# Types:
# GaussianNB: Continuous features (assumes Gaussian distribution)
# MultinomialNB: Discrete counts (text classification)
# BernoulliNB: Binary features
```

---

## 3. Unsupervised Learning

### K-Means Clustering

```python
from sklearn.cluster import KMeans

class KMeansScratch:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
    
    def fit(self, X):
        # Initialize centroids randomly
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            # Assign clusters
            distances = self._calc_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([
                X[self.labels == k].mean(axis=0) 
                for k in range(self.n_clusters)
            ])
            
            # Check convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids
        
        return self
    
    def _calc_distances(self, X):
        return np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
    
    def predict(self, X):
        distances = self._calc_distances(X)
        return np.argmin(distances, axis=1)

# Elbow method for optimal k
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Silhouette score
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)
```

### DBSCAN

```python
from sklearn.cluster import DBSCAN

# Density-Based Spatial Clustering
# - No need to specify number of clusters
# - Can find arbitrarily shaped clusters
# - Handles outliers as noise

dbscan = DBSCAN(
    eps=0.5,           # Maximum distance between points
    min_samples=5,     # Minimum points to form cluster
    metric='euclidean'
)

# Core points: >= min_samples within eps
# Border points: within eps of core point
# Noise: neither core nor border
```

### Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Agglomerative (bottom-up)
agg = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'  # 'single', 'complete', 'average', 'ward'
)

# Linkage methods:
# Single: min distance between clusters
# Complete: max distance between clusters
# Average: average distance between clusters
# Ward: minimize within-cluster variance

# Dendrogram
Z = linkage(X, method='ward')
dendrogram(Z)
```

### PCA (Principal Component Analysis)

```python
from sklearn.decomposition import PCA

class PCAScratch:
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.components = eigenvectors[:, idx][:, :self.n_components]
        
        # Explained variance ratio
        self.explained_variance_ratio = (
            self.eigenvalues[:self.n_components] / 
            np.sum(self.eigenvalues)
        )
        return self
    
    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Usage
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(f"Explained variance: {pca.explained_variance_ratio_}")
```

### t-SNE & UMAP

```python
from sklearn.manifold import TSNE
# pip install umap-learn
import umap

# t-SNE: t-distributed Stochastic Neighbor Embedding
tsne = TSNE(
    n_components=2,
    perplexity=30,     # Balance local vs global
    learning_rate=200,
    n_iter=1000
)
X_tsne = tsne.fit_transform(X)

# UMAP: Uniform Manifold Approximation and Projection
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1
)
X_umap = reducer.fit_transform(X)

# Key differences:
# - t-SNE: Better at preserving local structure
# - UMAP: Faster, preserves more global structure
# - Both are non-linear, good for visualization
```

---

## 4. Ensemble Methods

### Bagging (Bootstrap Aggregating)

```python
from sklearn.ensemble import BaggingClassifier

# Key idea: Train multiple models on bootstrap samples
# Reduces variance, prevents overfitting

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,      # Fraction of samples
    max_features=0.8,     # Fraction of features
    bootstrap=True,
    oob_score=True
)
```

### Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb

# AdaBoost: Adaptive Boosting
# - Adjust sample weights based on errors
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=0.1
)

# Gradient Boosting
# - Fit new model to residuals
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8
)

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,          # L1 regularization
    reg_lambda=1,         # L2 regularization
    use_label_encoder=False,
    eval_metric='logloss'
)

# LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8
)

# Key differences:
# XGBoost: Level-wise tree growth, more regularization options
# LightGBM: Leaf-wise growth, faster, histogram-based
```

### Stacking

```python
from sklearn.ensemble import StackingClassifier

# Combine multiple models with a meta-learner
stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier()),
        ('svm', SVC(probability=True))
    ],
    final_estimator=LogisticRegression(),
    cv=5
)
```

---

## 5. Model Evaluation

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)

# Accuracy: (TP + TN) / (TP + TN + FP + FN)
accuracy = accuracy_score(y_true, y_pred)

# Precision: TP / (TP + FP) - "Of predicted positives, how many are correct?"
precision = precision_score(y_true, y_pred)

# Recall/Sensitivity: TP / (TP + FN) - "Of actual positives, how many did we find?"
recall = recall_score(y_true, y_pred)

# F1 Score: 2 * (precision * recall) / (precision + recall)
f1 = f1_score(y_true, y_pred)

# Confusion Matrix
#              Predicted
#              Neg    Pos
# Actual Neg   TN     FP
#        Pos   FN     TP
cm = confusion_matrix(y_true, y_pred)

# ROC-AUC
auc = roc_auc_score(y_true, y_proba)
fpr, tpr, thresholds = roc_curve(y_true, y_proba)

# Precision-Recall Curve (better for imbalanced data)
precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_proba)
ap = average_precision_score(y_true, y_proba)
```

### Regression Metrics

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

# MSE: Mean Squared Error
mse = mean_squared_error(y_true, y_pred)

# RMSE: Root Mean Squared Error
rmse = np.sqrt(mse)

# MAE: Mean Absolute Error
mae = mean_absolute_error(y_true, y_pred)

# R² Score: 1 - (SS_res / SS_tot)
r2 = r2_score(y_true, y_pred)

# MAPE: Mean Absolute Percentage Error
mape = mean_absolute_percentage_error(y_true, y_pred)
```

### Cross-Validation

```python
from sklearn.model_selection import (
    cross_val_score, cross_validate, KFold, StratifiedKFold,
    TimeSeriesSplit, LeaveOneOut
)

# K-Fold Cross-Validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Stratified K-Fold (maintains class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Time Series Split
tscv = TimeSeriesSplit(n_splits=5)

# Multiple metrics
results = cross_validate(
    model, X, y, cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1']
)
```

---

## 6. Feature Engineering

### Handling Missing Data

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Simple imputation
imputer = SimpleImputer(strategy='mean')  # 'median', 'most_frequent', 'constant'
X_imputed = imputer.fit_transform(X)

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(X)

# Detection
missing_pct = df.isnull().sum() / len(df) * 100
```

### Encoding Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# Label Encoding (for ordinal categories)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-Hot Encoding (for nominal categories)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = ohe.fit_transform(X[['category_column']])

# Pandas get_dummies
df_encoded = pd.get_dummies(df, columns=['category'], drop_first=True)

# Target Encoding
def target_encode(df, column, target):
    means = df.groupby(column)[target].mean()
    return df[column].map(means)
```

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standardization (Z-score): (x - mean) / std
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-Max Scaling: (x - min) / (max - min)
minmax = MinMaxScaler()
X_scaled = minmax.fit_transform(X)

# Robust Scaling (handles outliers)
robust = RobustScaler()
X_scaled = robust.fit_transform(X)

# When to use what:
# StandardScaler: Algorithms assuming Gaussian distribution
# MinMaxScaler: Neural networks, algorithms with bounded inputs
# RobustScaler: Data with outliers
```

### Feature Selection

```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)

# Filter Methods
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Wrapper Methods (RFE)
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

# Embedded Methods
selector = SelectFromModel(RandomForestClassifier(), threshold='median')
X_selected = selector.fit_transform(X, y)

# Variance Threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.1)
X_selected = selector.fit_transform(X)
```

---

## 7. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X, y)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

# Random Search (more efficient for large search spaces)
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Bayesian Optimization (optuna)
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
    }
    model = RandomForestClassifier(**params)
    return cross_val_score(model, X, y, cv=5).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(f"Best params: {study.best_params}")
```

---

## 8. Handling Imbalanced Data

```python
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Combination
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

# Class weights
model = RandomForestClassifier(class_weight='balanced')

# Threshold adjustment
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.3).astype(int)  # Lower threshold
```

---

## 9. Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Error

High Bias (Underfitting):
- Model too simple
- High training error
- High test error
- Solutions: More features, complex model, less regularization

High Variance (Overfitting):
- Model too complex
- Low training error
- High test error
- Solutions: More data, simpler model, regularization, dropout

Sweet Spot:
- Balance between bias and variance
- Good generalization
- Similar training and test error
```

---

## 10. ML System Design Checklist

1. **Problem Framing**
   - What's the business goal?
   - What type of ML problem? (classification, regression, etc.)
   - What are success metrics?

2. **Data**
   - What data is available?
   - How to collect/label data?
   - Data quality and preprocessing

3. **Features**
   - Feature engineering
   - Feature selection
   - Feature storage (feature store)

4. **Model**
   - Model selection
   - Training pipeline
   - Hyperparameter tuning

5. **Evaluation**
   - Offline metrics
   - Online metrics (A/B testing)
   - Error analysis

6. **Deployment**
   - Batch vs real-time
   - Model serving
   - Monitoring and alerting

7. **Maintenance**
   - Model retraining
   - Data drift detection
   - Model versioning

# Supervised Learning - Complete Guide

## ⚡ Interview Quick Summary

> **Core insight**: Supervised learning finds a function f(x) ≈ y from labeled examples. The entire field is about the bias-variance tradeoff: how complex should f be, and how do we prevent it from memorizing noise?

### Algorithm Selection Guide

| Algorithm | Best When | Strength | Weakness |
|-----------|-----------|----------|----------|
| Linear/Logistic Regression | Linear relationships, need interpretability | Fast, calibrated | Can't capture non-linearity |
| Decision Trees | Non-linear, categorical features, interpretability needed | Visual, handles mixed types | High variance, overfits |
| Random Forest | General purpose, good baseline | Low variance, handles missing | Slow inference, black box |
| XGBoost / LightGBM | Tabular data competition | Best for tabular, handles NaN | Slow to tune |
| SVM | Small dataset, high-dim features | Max margin, kernel trick | Slow O(n²-n³), needs scaling |
| KNN | Baseline, anomaly detection | Simple, no training | O(n) inference, curse of dim |
| Naive Bayes | Text classification | Fast, good for NLP | Assumes feature independence |
| Neural Networks | Complex patterns, images, text | Universal approximator | Needs lots of data, slow |

### Key Loss Functions — Know Which to Use When

```
Regression:
  MSE:   L = (y - ŷ)²         ← penalizes outliers heavily (squared)
  MAE:   L = |y - ŷ|          ← robust to outliers (linear)
  Huber: L = MSE if |δ|<δ, MAE otherwise  ← best of both

Classification:
  Cross-entropy: L = -Σ y_i log(ŷ_i)   ← measures distribution mismatch
  Binary CE:     L = -[y log(p) + (1-y) log(1-p)]  ← logistic regression
  Focal loss:    L = -α(1-p)^γ log(p)  ← for imbalanced (down-weight easy)

Regularized:
  Ridge: MSE + λ||w||²   ← L2, shrinks weights, closed-form solution
  Lasso: MSE + λ||w||    ← L1, sparse weights, feature selection
```

### 🚨 Top Interview Pitfalls
- Recommending neural networks for tabular data before trying **XGBoost/LightGBM** (which usually wins on tabular)
- Forgetting to **scale features** before SVM, KNN, or neural networks (tree-based methods don't need scaling)
- Using MSE for classification or cross-entropy for regression (always match loss to task)
- Not mentioning **class_weight='balanced'** or threshold tuning for imbalanced classification

---

## Table of Contents
1. [Regression Algorithms](#regression-algorithms)
2. [Classification Algorithms](#classification-algorithms)
3. [Tree-Based Methods](#tree-based-methods)
4. [Support Vector Machines](#support-vector-machines)
5. [Ensemble Methods](#ensemble-methods)
6. [Interview Questions](#interview-questions)

---

## 1. Regression Algorithms

### Linear Regression

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# From Scratch - Normal Equation
class LinearRegressionNormalEquation:
    """
    Closed-form solution: θ = (X^T X)^(-1) X^T y
    Time: O(n³) for matrix inversion
    """
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        self.theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta


# From Scratch - Gradient Descent
class LinearRegressionGD:
    """
    Iterative solution using gradient descent.
    Loss: MSE = (1/2n) * Σ(y - ŷ)²
    Gradient: ∂MSE/∂w = (1/n) * X^T(ŷ - y)
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, tol=1e-6):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.tol = tol
        self.loss_history = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iters):
            # Forward pass
            y_pred = X @ self.weights + self.bias
            
            # Compute loss
            loss = np.mean((y - y_pred) ** 2) / 2
            self.loss_history.append(loss)
            
            # Compute gradients
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Early stopping
            if i > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                break
        
        return self
    
    def predict(self, X):
        return X @ self.weights + self.bias


# Regularized Linear Regression
"""
Ridge (L2): Loss + λ||w||²
- Shrinks coefficients toward zero
- Handles multicollinearity
- Closed-form solution exists

Lasso (L1): Loss + λ||w||₁
- Can set coefficients exactly to zero (feature selection)
- Sparse solutions
- No closed-form solution

ElasticNet: Loss + λ₁||w||₁ + λ₂||w||²
- Combines benefits of both
- Good when features are correlated
"""

# Ridge Regression from Scratch
class RidgeRegressionScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def fit(self, X, y):
        n_features = X.shape[1]
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # θ = (X^T X + αI)^(-1) X^T y
        I = np.eye(n_features + 1)
        I[0, 0] = 0  # Don't regularize bias
        self.theta = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
        return self
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta


# Sklearn usage
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Always scale features for regularized models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Polynomial Regression
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)
model = Ridge(alpha=0.1)
model.fit(X_poly, y)
```

### Logistic Regression

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

class LogisticRegressionScratch:
    """
    Binary classification using sigmoid function.
    P(y=1|x) = σ(wᵀx + b) = 1 / (1 + e^(-(wᵀx + b)))
    
    Loss: Binary Cross-Entropy
    L = -1/n Σ [y·log(p) + (1-y)·log(1-p)]
    
    Gradient:
    ∂L/∂w = 1/n Σ (p - y)x
    ∂L/∂b = 1/n Σ (p - y)
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, 
                 regularization='l2', C=1.0):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.reg = regularization
        self.C = C  # Inverse regularization strength
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)
            
            # Gradients
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Add regularization gradient
            if self.reg == 'l2':
                dw += (1 / self.C) * self.weights
            elif self.reg == 'l1':
                dw += (1 / self.C) * np.sign(self.weights)
            
            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
        return self
    
    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        probs = self.sigmoid(z)
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# Multi-class Logistic Regression (Softmax)
class SoftmaxRegression:
    """
    Multi-class classification using softmax.
    P(y=k|x) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx)
    
    Loss: Categorical Cross-Entropy
    L = -1/n Σᵢ Σₖ yᵢₖ·log(pᵢₖ)
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iters = n_iterations
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # One-hot encode y
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y] = 1
        
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)
        
        for _ in range(self.n_iters):
            z = X @ self.weights + self.bias
            y_pred = self.softmax(z)
            
            # Gradients
            dw = (1 / n_samples) * X.T @ (y_pred - y_onehot)
            db = (1 / n_samples) * np.sum(y_pred - y_onehot, axis=0)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
        return self
    
    def predict(self, X):
        z = X @ self.weights + self.bias
        return np.argmax(self.softmax(z), axis=1)


# Sklearn multi-class strategies
log_reg = LogisticRegression(
    multi_class='multinomial',  # or 'ovr' (one-vs-rest)
    solver='lbfgs',
    C=1.0,
    max_iter=1000
)
```

---

## 2. Classification Algorithms

### K-Nearest Neighbors (KNN)

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from collections import Counter

class KNNClassifierScratch:
    """
    Non-parametric, instance-based learning.
    Prediction: majority vote of k nearest neighbors.
    """
    
    def __init__(self, k=5, distance_metric='euclidean', weights='uniform'):
        self.k = k
        self.metric = distance_metric
        self.weights = weights
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _compute_distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'minkowski':
            p = 3
            return np.sum(np.abs(x1 - x2) ** p) ** (1/p)
    
    def _get_neighbors(self, x):
        distances = [self._compute_distance(x, x_train) 
                     for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_distances = np.array(distances)[k_indices]
        k_labels = self.y_train[k_indices]
        return k_indices, k_distances, k_labels
    
    def predict(self, X):
        predictions = []
        for x in X:
            _, k_distances, k_labels = self._get_neighbors(x)
            
            if self.weights == 'uniform':
                # Simple majority vote
                most_common = Counter(k_labels).most_common(1)[0][0]
            elif self.weights == 'distance':
                # Weighted by inverse distance
                weights = 1 / (k_distances + 1e-10)
                weighted_votes = {}
                for label, weight in zip(k_labels, weights):
                    weighted_votes[label] = weighted_votes.get(label, 0) + weight
                most_common = max(weighted_votes, key=weighted_votes.get)
            
            predictions.append(most_common)
        return np.array(predictions)


# KNN Regressor
class KNNRegressorScratch:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_values = self.y_train[k_indices]
            predictions.append(np.mean(k_values))
        return np.array(predictions)


# Choosing optimal k
from sklearn.model_selection import cross_val_score

k_values = range(1, 31)
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
```

### Naive Bayes

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

class GaussianNBScratch:
    """
    Assumes features follow Gaussian distribution.
    P(xᵢ|y) = (1/√(2πσ²)) exp(-(xᵢ-μ)²/(2σ²))
    
    Naive assumption: features are conditionally independent given class.
    P(x₁, x₂, ..., xₙ|y) = ∏ P(xᵢ|y)
    
    Prediction: argmax_y P(y) ∏ P(xᵢ|y)
    """
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        
        # Calculate class statistics
        self.class_prior = {}
        self.mean = {}
        self.var = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.class_prior[c] = len(X_c) / len(X)
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + 1e-9  # Add smoothing
        
        return self
    
    def _gaussian_pdf(self, x, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * \
               np.exp(-(x - mean)**2 / (2 * var))
    
    def _calculate_posterior(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.class_prior[c])
            likelihood = np.sum(np.log(
                self._gaussian_pdf(x, self.mean[c], self.var[c])
            ))
            posteriors.append(prior + likelihood)
        return posteriors
    
    def predict(self, X):
        return np.array([
            self.classes[np.argmax(self._calculate_posterior(x))]
            for x in X
        ])
    
    def predict_proba(self, X):
        probs = []
        for x in X:
            posteriors = np.array(self._calculate_posterior(x))
            # Convert log probabilities to probabilities
            posteriors = np.exp(posteriors - np.max(posteriors))
            posteriors /= np.sum(posteriors)
            probs.append(posteriors)
        return np.array(probs)


# Multinomial Naive Bayes (for text classification)
class MultinomialNBScratch:
    """
    For discrete count data (e.g., word counts).
    P(xᵢ|y) = (Nᵧᵢ + α) / (Nᵧ + α·n)
    α: Laplace smoothing parameter
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_prior = {}
        self.feature_prob = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.class_prior[c] = len(X_c) / len(X)
            
            # Feature probabilities with Laplace smoothing
            feature_counts = X_c.sum(axis=0) + self.alpha
            total_counts = feature_counts.sum()
            self.feature_prob[c] = feature_counts / total_counts
        
        return self
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.class_prior[c])
                likelihood = np.sum(x * np.log(self.feature_prob[c]))
                posteriors.append(prior + likelihood)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)


# When to use which Naive Bayes:
# GaussianNB: Continuous features (assumed Gaussian)
# MultinomialNB: Discrete counts (text classification, TF-IDF)
# BernoulliNB: Binary features (document contains word or not)
```

---

## 3. Tree-Based Methods

### Decision Trees

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class DecisionTreeClassifierScratch:
    """
    Recursive partitioning of feature space.
    Split criteria: Information Gain, Gini Impurity, or MSE (regression)
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree = None
    
    def gini(self, y):
        """Gini Impurity: 1 - Σ pᵢ²"""
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def entropy(self, y):
        """Entropy: -Σ pᵢ log₂(pᵢ)"""
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def information_gain(self, parent, left, right):
        """IG = H(parent) - weighted_avg(H(children))"""
        n = len(parent)
        n_left, n_right = len(left), len(right)
        
        if self.criterion == 'gini':
            impurity_func = self.gini
        else:
            impurity_func = self.entropy
        
        parent_impurity = impurity_func(parent)
        child_impurity = (n_left/n * impurity_func(left) + 
                          n_right/n * impurity_func(right))
        
        return parent_impurity - child_impurity
    
    def best_split(self, X, y):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if (np.sum(left_mask) < self.min_samples_leaf or 
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue
                
                gain = self.information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth or
            n_classes == 1 or
            n_samples < self.min_samples_split):
            # Leaf node
            return {'leaf': True, 'value': np.bincount(y).argmax()}
        
        # Find best split
        feature, threshold, gain = self.best_split(X, y)
        
        if feature is None:
            return {'leaf': True, 'value': np.bincount(y).argmax()}
        
        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # Recursively build subtrees
        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self.build_tree(X, y)
        return self
    
    def _predict_sample(self, x, node):
        if node['leaf']:
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])


# Decision Tree for Regression
class DecisionTreeRegressorScratch:
    """Uses MSE as split criterion, predicts mean of leaf values."""
    
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    def mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)
    
    def best_split(self, X, y):
        best_mse = np.inf
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                mse_left = self.mse(y[left_mask])
                mse_right = self.mse(y[right_mask])
                weighted_mse = (len(y[left_mask]) * mse_left + 
                               len(y[right_mask]) * mse_right) / len(y)
                
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    # ... similar build_tree and predict methods
```

### Random Forest

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class RandomForestClassifierScratch:
    """
    Ensemble of decision trees trained on bootstrap samples.
    
    Key ideas:
    1. Bagging: Bootstrap samples for each tree
    2. Random feature subset at each split
    3. Aggregate predictions: majority vote (classification) / mean (regression)
    """
    
    def __init__(self, n_estimators=100, max_depth=None, 
                 min_samples_split=2, max_features='sqrt', bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []
        self.feature_indices = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Determine number of features to consider at each split
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
            else:
                indices = np.arange(n_samples)
            
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Random feature selection
            feature_idx = np.random.choice(n_features, max_features, replace=False)
            
            # Train tree on subset of features
            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_bootstrap[:, feature_idx], y_bootstrap)
            
            self.trees.append(tree)
            self.feature_indices.append(feature_idx)
        
        return self
    
    def predict(self, X):
        # Collect predictions from all trees
        predictions = np.array([
            tree.predict(X[:, self.feature_indices[i]])
            for i, tree in enumerate(self.trees)
        ])
        
        # Majority vote
        return np.array([
            np.bincount(predictions[:, i]).argmax()
            for i in range(X.shape[0])
        ])
    
    def feature_importances_(self, X, y):
        """Calculate feature importance via permutation."""
        baseline = self.score(X, y)
        importances = np.zeros(X.shape[1])
        
        for feature in range(X.shape[1]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feature])
            permuted_score = self.score(X_permuted, y)
            importances[feature] = baseline - permuted_score
        
        return importances / np.sum(importances)


# Out-of-Bag (OOB) Score
"""
For each tree, ~37% of samples are not used (out-of-bag).
OOB score: use these samples to estimate generalization error.
No need for separate validation set!
"""

rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X, y)
print(f"OOB Score: {rf.oob_score_}")
```

---

## 4. Support Vector Machines

```python
import numpy as np
from sklearn.svm import SVC, SVR

"""
SVM: Find hyperplane that maximizes margin between classes.

Hard margin (linearly separable):
min ||w||² / 2
s.t. yᵢ(wᵀxᵢ + b) ≥ 1

Soft margin (allow some misclassification):
min ||w||² / 2 + C Σ ξᵢ
s.t. yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ
     ξᵢ ≥ 0

C: regularization parameter
- Large C: smaller margin, fewer violations
- Small C: larger margin, more violations
"""

# Kernel Trick
"""
Map data to higher dimensions to find linear separator.
K(x, y) = φ(x)ᵀφ(y)  (inner product in transformed space)

Common kernels:
- Linear: K(x, y) = xᵀy
- Polynomial: K(x, y) = (γxᵀy + r)^d
- RBF/Gaussian: K(x, y) = exp(-γ||x-y||²)
- Sigmoid: K(x, y) = tanh(γxᵀy + r)
"""

# Linear SVM from Scratch (simplified)
class LinearSVMScratch:
    def __init__(self, C=1.0, learning_rate=0.001, n_iters=1000):
        self.C = C
        self.lr = learning_rate
        self.n_iters = n_iters
    
    def fit(self, X, y):
        # Convert labels to -1, 1
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            for i, x in enumerate(X):
                # Check if correctly classified with margin
                if y_[i] * (np.dot(x, self.weights) + self.bias) >= 1:
                    # Correctly classified, only regularization gradient
                    self.weights -= self.lr * (2 * self.weights / self.C)
                else:
                    # Misclassified, hinge loss gradient
                    self.weights -= self.lr * (
                        2 * self.weights / self.C - y_[i] * x
                    )
                    self.bias -= self.lr * (-y_[i])
        
        return self
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)


# Sklearn SVM usage
svm = SVC(
    C=1.0,              # Regularization (inverse)
    kernel='rbf',       # 'linear', 'poly', 'rbf', 'sigmoid'
    gamma='scale',      # Kernel coefficient ('scale' or 'auto' or float)
    degree=3,           # Polynomial degree
    probability=True,   # Enable probability estimates
    class_weight='balanced'  # Handle imbalanced classes
)

# Multi-class SVM
# One-vs-Rest (OvR): Train K classifiers
# One-vs-One (OvO): Train K(K-1)/2 classifiers
svm_ovr = SVC(decision_function_shape='ovr')
svm_ovo = SVC(decision_function_shape='ovo')

# Support Vector Regression
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
# epsilon: margin of tolerance (tube around prediction)
```

---

## 5. Ensemble Methods

### Bagging

```python
from sklearn.ensemble import BaggingClassifier, BaggingRegressor

"""
Bagging (Bootstrap Aggregating):
1. Create bootstrap samples (sample with replacement)
2. Train model on each sample
3. Aggregate predictions (vote/average)

Reduces variance, prevents overfitting.
Works best with high-variance models (e.g., deep trees).
"""

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None),
    n_estimators=100,
    max_samples=1.0,      # Fraction of samples for each tree
    max_features=1.0,     # Fraction of features for each tree
    bootstrap=True,       # Sample with replacement
    bootstrap_features=False,
    oob_score=True,
    n_jobs=-1
)
```

### Boosting

```python
import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier,
    GradientBoostingRegressor
)
import xgboost as xgb
import lightgbm as lgb

# AdaBoost from Scratch
class AdaBoostClassifierScratch:
    """
    Adaptive Boosting:
    1. Initialize equal sample weights
    2. Train weak learner
    3. Calculate error and learner weight
    4. Update sample weights (increase misclassified)
    5. Repeat
    
    Final prediction: weighted vote of all learners
    """
    
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.alphas = []
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        y_ = np.where(y == 0, -1, 1)  # Convert to -1, 1
        
        # Initialize sample weights
        sample_weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            # Train weak learner (decision stump)
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=sample_weights)
            
            # Make predictions
            predictions = stump.predict(X)
            predictions = np.where(predictions == 0, -1, 1)
            
            # Calculate weighted error
            misclassified = predictions != y_
            error = np.sum(sample_weights * misclassified) / np.sum(sample_weights)
            error = np.clip(error, 1e-10, 1 - 1e-10)
            
            # Calculate learner weight
            alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)
            
            # Update sample weights
            sample_weights *= np.exp(-alpha * y_ * predictions)
            sample_weights /= np.sum(sample_weights)
            
            self.estimators.append(stump)
            self.alphas.append(alpha)
        
        return self
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        
        for alpha, stump in zip(self.alphas, self.estimators):
            pred = stump.predict(X)
            pred = np.where(pred == 0, -1, 1)
            predictions += alpha * pred
        
        return (np.sign(predictions) + 1) // 2  # Convert back to 0, 1


# Gradient Boosting
"""
Gradient Boosting:
1. Initialize with constant prediction
2. Calculate residuals (negative gradients)
3. Fit new tree to residuals
4. Update predictions
5. Repeat

Each tree corrects errors of previous ensemble.
"""

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,       # Shrinkage
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=0.8,           # Row subsampling (Stochastic GB)
    max_features='sqrt',     # Column subsampling
    loss='log_loss',         # 'log_loss', 'exponential'
    validation_fraction=0.1,
    n_iter_no_change=10      # Early stopping
)


# XGBoost
"""
Extreme Gradient Boosting:
- Regularized objective: Loss + Ω(complexity)
- Second-order Taylor approximation of loss
- Efficient tree construction (histogram-based)
- Handles missing values
- Parallel processing
"""

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,      # Min sum of instance weight in child
    subsample=0.8,           # Row sampling
    colsample_bytree=0.8,    # Column sampling per tree
    colsample_bylevel=1.0,   # Column sampling per level
    reg_alpha=0,             # L1 regularization
    reg_lambda=1,            # L2 regularization
    gamma=0,                 # Min loss reduction for split
    scale_pos_weight=1,      # Balance positive/negative weights
    objective='binary:logistic',
    eval_metric='logloss',
    early_stopping_rounds=10,
    use_label_encoder=False
)

# Training with validation
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)


# LightGBM
"""
Light Gradient Boosting Machine:
- Leaf-wise tree growth (vs level-wise in XGBoost)
- Gradient-based One-Side Sampling (GOSS)
- Exclusive Feature Bundling (EFB)
- Faster and more memory efficient
"""

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,            # No limit
    num_leaves=31,           # Max leaves in one tree
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=0,
    importance_type='gain'   # or 'split'
)


# CatBoost (handles categorical features)
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3,
    cat_features=['cat_col1', 'cat_col2'],  # Categorical column names
    verbose=False
)
```

### Stacking

```python
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression

"""
Stacking:
1. Train base learners on training data
2. Use base learners to make predictions (meta-features)
3. Train meta-learner on meta-features
4. Final prediction: meta-learner on base predictions
"""

stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('gb', GradientBoostingClassifier(n_estimators=100)),
        ('svm', SVC(probability=True)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ],
    final_estimator=LogisticRegression(),
    cv=5,                    # Cross-validation for meta-features
    stack_method='auto',     # 'auto', 'predict_proba', 'decision_function', 'predict'
    passthrough=False        # Include original features with meta-features
)
```

---

## 6. Interview Questions

```python
# Q1: When to use logistic regression vs decision trees vs SVM?
"""
Logistic Regression:
- Linear decision boundary
- Interpretable coefficients
- Works well with many features
- Needs feature scaling

Decision Trees:
- Non-linear boundaries
- Handles categorical features naturally
- Feature importance built-in
- Prone to overfitting

SVM:
- Works well in high dimensions
- Memory efficient (only support vectors)
- Kernel trick for non-linear boundaries
- Doesn't scale well with large datasets
"""


# Q2: Explain bias-variance tradeoff in ensemble methods
"""
Bagging (Random Forest):
- Reduces variance by averaging
- Bias stays similar to individual trees
- Best for high-variance models (deep trees)

Boosting (XGBoost):
- Reduces bias by iteratively fitting residuals
- Can increase variance
- Best for high-bias models (shallow trees)
"""


# Q3: Why does Random Forest use max_features < total features?
"""
1. De-correlation: Prevents all trees from being similar
2. Different trees capture different patterns
3. Improves ensemble diversity
4. sqrt(n_features) is common default
"""


# Q4: How does XGBoost handle overfitting?
"""
1. Regularization: L1 (alpha) and L2 (lambda) penalties
2. Shrinkage: learning_rate reduces each tree's contribution
3. Column/row subsampling: randomness in training
4. Early stopping: stop when validation score stops improving
5. Max depth limit: controls tree complexity
6. Minimum child weight: prevents splits with few samples
"""


# Q5: Explain the kernel trick in SVM
"""
Instead of explicitly computing φ(x) in high-dimensional space,
we compute K(x, y) = φ(x)·φ(y) directly.

RBF kernel: K(x, y) = exp(-γ||x-y||²)
- Equivalent to infinite-dimensional feature space
- No need to compute explicit transformation
- Only inner products needed for optimization
"""


# Q6: What's the difference between Gini and Entropy?
"""
Gini: 1 - Σ pᵢ²
- Faster to compute (no logarithm)
- Tends to isolate most frequent class

Entropy: -Σ pᵢ log₂(pᵢ)
- Information theoretic measure
- Tends to produce more balanced trees

In practice: similar results
"""


# Q7: Why do we need feature scaling for some algorithms?
"""
Distance-based algorithms (KNN, SVM, K-Means):
- Features on different scales dominate distance calculation

Gradient-based optimization (Logistic Regression, Neural Networks):
- Different scales cause different gradient magnitudes
- Slow convergence, difficulty finding optimal learning rate

Tree-based algorithms (Random Forest, XGBoost):
- Don't need scaling (split based on thresholds)
"""


# Q8: How to choose between L1 and L2 regularization?
"""
L1 (Lasso):
- Produces sparse solutions (feature selection)
- When you suspect many features are irrelevant
- When interpretability with fewer features is important

L2 (Ridge):
- When all features are relevant
- When features are correlated
- When you want smooth coefficient distribution

ElasticNet: When unsure, combines both
"""


# Q9: What is the Naive Bayes assumption and when does it fail?
"""
Assumption: Features are conditionally independent given the class
P(x₁, x₂, ..., xₙ|y) = ∏ P(xᵢ|y)

Fails when:
- Features are correlated (e.g., height and weight)
- Features have complex interactions
- Feature dependencies are crucial for classification

Still works well in practice for text classification
because it estimates decision boundary, not true probabilities.
"""


# Q10: Explain out-of-bag (OOB) error in Random Forest
"""
For each bootstrap sample, ~37% of data is not used (out-of-bag).

OOB Error:
1. For each sample, collect predictions from trees that didn't train on it
2. Aggregate these predictions (majority vote)
3. Compare with true labels

Benefits:
- Free validation estimate
- No need for separate validation set
- Unbiased estimate of generalization error
"""
```

# Model Evaluation & Selection - Complete Guide

## Table of Contents
1. [Classification Metrics](#classification-metrics)
2. [Regression Metrics](#regression-metrics)
3. [Cross-Validation](#cross-validation)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Bias-Variance Tradeoff](#bias-variance-tradeoff)
6. [Interview Questions](#interview-questions)

---

## 1. Classification Metrics

### Confusion Matrix

```python
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

"""
Confusion Matrix:
                    Predicted
                    Neg     Pos
Actual  Neg         TN      FP      (Type I Error)
        Pos         FN      TP      (Type II Error)

TN: True Negative  - Correctly predicted negative
FP: False Positive - Incorrectly predicted positive (Type I)
FN: False Negative - Incorrectly predicted negative (Type II)
TP: True Positive  - Correctly predicted positive
"""

y_true = [0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 1, 0, 0, 1, 1, 1]

cm = confusion_matrix(y_true, y_pred)
# [[2, 1],
#  [2, 3]]

tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


# Visualize confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
```

### Core Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score
)

"""
Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Overall correctness
- Misleading for imbalanced data

Precision = TP / (TP + FP)
- "Of predicted positives, how many are correct?"
- Use when false positives are costly (spam detection)

Recall (Sensitivity, TPR) = TP / (TP + FN)
- "Of actual positives, how many did we find?"
- Use when false negatives are costly (disease detection)

Specificity (TNR) = TN / (TN + FP)
- "Of actual negatives, how many did we correctly identify?"

F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
- Harmonic mean of precision and recall
- Good for imbalanced datasets
"""

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
specificity = tn / (tn + fp)

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"Specificity: {specificity:.3f}")


# F-beta Score (weighted F1)
from sklearn.metrics import fbeta_score

# β > 1: emphasize recall
# β < 1: emphasize precision
# β = 1: standard F1
f2 = fbeta_score(y_true, y_pred, beta=2)  # Emphasize recall


# Balanced Accuracy (useful for imbalanced data)
# (TPR + TNR) / 2
balanced_acc = balanced_accuracy_score(y_true, y_pred)


# Classification Report (all metrics at once)
print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
```

### Multi-Class Metrics

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report
)

y_true_multi = [0, 0, 1, 1, 2, 2, 2]
y_pred_multi = [0, 1, 1, 1, 2, 0, 2]

# Averaging strategies
"""
macro: Calculate metric for each class, then average (treats all classes equally)
weighted: Average weighted by class support (number of samples)
micro: Calculate global metric (total TP, FP, FN)
"""

precision_macro = precision_score(y_true_multi, y_pred_multi, average='macro')
precision_weighted = precision_score(y_true_multi, y_pred_multi, average='weighted')
precision_micro = precision_score(y_true_multi, y_pred_multi, average='micro')

# Per-class metrics
precision_per_class = precision_score(y_true_multi, y_pred_multi, average=None)

print(classification_report(y_true_multi, y_pred_multi))
```

### ROC Curve and AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

"""
ROC Curve (Receiver Operating Characteristic):
- Plots TPR (recall) vs FPR (1 - specificity) at various thresholds
- FPR = FP / (FP + TN) = 1 - Specificity

AUC (Area Under ROC Curve):
- 1.0 = perfect classifier
- 0.5 = random classifier
- < 0.5 = worse than random (predictions inverted)

Useful for:
- Comparing models
- Threshold-independent evaluation
- Balanced class problems
"""

# Need probability predictions
y_true = np.array([0, 0, 0, 1, 1, 1])
y_proba = np.array([0.1, 0.3, 0.4, 0.6, 0.7, 0.9])

fpr, tpr, thresholds = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)
# or
roc_auc = roc_auc_score(y_true, y_proba)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# Multi-class ROC-AUC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# One-vs-Rest
y_true_bin = label_binarize(y_true_multi, classes=[0, 1, 2])
y_proba_multi = model.predict_proba(X_test)

roc_auc_ovr = roc_auc_score(y_true_bin, y_proba_multi, multi_class='ovr')
roc_auc_ovo = roc_auc_score(y_true_bin, y_proba_multi, multi_class='ovo')
```

### Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

"""
PR Curve: Plots Precision vs Recall at various thresholds

Useful for:
- Imbalanced datasets (preferred over ROC)
- When positive class is more important
- When false positives and false negatives have different costs

AP (Average Precision): Area under PR curve
"""

precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
ap = average_precision_score(y_true, y_proba)

# Plot PR curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()


# Finding optimal threshold
def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """Find threshold that maximizes chosen metric."""
    thresholds = np.arange(0, 1, 0.01)
    scores = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        if metric == 'f1':
            scores.append(f1_score(y_true, y_pred))
        elif metric == 'precision':
            scores.append(precision_score(y_true, y_pred))
        elif metric == 'recall':
            scores.append(recall_score(y_true, y_pred))
    
    optimal_idx = np.argmax(scores)
    return thresholds[optimal_idx], scores[optimal_idx]
```

### Log Loss and Calibration

```python
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

"""
Log Loss (Cross-Entropy Loss):
- Penalizes confident wrong predictions heavily
- Lower is better
- L = -1/n Σ [y*log(p) + (1-y)*log(1-p)]

Brier Score:
- Mean squared error of probability predictions
- Lower is better
- Range: [0, 1]
"""

log_loss_score = log_loss(y_true, y_proba)
brier_score = brier_score_loss(y_true, y_proba)


# Probability Calibration
"""
Well-calibrated: predicted probabilities match true frequencies
E.g., of predictions with P=0.8, ~80% should be positive
"""

# Calibration curve (reliability diagram)
fraction_of_positives, mean_predicted = calibration_curve(
    y_true, y_proba, n_bins=10
)

plt.figure(figsize=(8, 6))
plt.plot(mean_predicted, fraction_of_positives, 's-', label='Model')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.show()


# Calibrate a classifier
calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)
calibrated_proba = calibrated_model.predict_proba(X_test)
```

---

## 2. Regression Metrics

```python
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)

y_true = np.array([3, 5, 2.5, 7])
y_pred = np.array([2.5, 5, 4, 8])

# Mean Squared Error (MSE)
# MSE = 1/n Σ(y - ŷ)²
# - Penalizes large errors more than small ones
# - Same units as y²
mse = mean_squared_error(y_true, y_pred)

# Root Mean Squared Error (RMSE)
# RMSE = √MSE
# - Same units as y
# - More interpretable than MSE
rmse = np.sqrt(mse)

# Mean Absolute Error (MAE)
# MAE = 1/n Σ|y - ŷ|
# - Less sensitive to outliers than MSE
# - Same units as y
mae = mean_absolute_error(y_true, y_pred)

# R² Score (Coefficient of Determination)
# R² = 1 - SS_res/SS_tot = 1 - Σ(y-ŷ)²/Σ(y-ȳ)²
# - Proportion of variance explained by model
# - 1 = perfect, 0 = as good as mean, <0 = worse than mean
r2 = r2_score(y_true, y_pred)

# Adjusted R² (penalizes additional features)
# R²_adj = 1 - (1-R²)(n-1)/(n-p-1)
# n = samples, p = features
def adjusted_r2(r2, n_samples, n_features):
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

# Mean Absolute Percentage Error (MAPE)
# MAPE = 100/n Σ|y - ŷ|/|y|
# - Scale-independent (percentage)
# - Undefined when y = 0
mape = mean_absolute_percentage_error(y_true, y_pred)

# Symmetric MAPE (handles zeros better)
# sMAPE = 100/n Σ |y - ŷ| / ((|y| + |ŷ|)/2)
def symmetric_mape(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / 
                   (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100

# Explained Variance
# Similar to R² but doesn't account for bias
explained_var = explained_variance_score(y_true, y_pred)


print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.2f}%")
```

### Choosing Regression Metrics

```python
"""
When to use which metric:

MSE/RMSE:
- When large errors are particularly bad
- When outliers should be penalized
- Most common choice

MAE:
- When outliers shouldn't dominate
- More robust to outliers
- Easier to interpret

MAPE:
- When relative error matters
- Comparing across different scales
- Avoid when y can be zero

R²:
- Comparing models
- Understanding variance explained
- Can be misleading (always compare with baseline)

Consider multiple metrics for complete picture.
"""

# Residual analysis
def residual_analysis(y_true, y_pred):
    residuals = y_true - y_pred
    
    # Residual statistics
    print(f"Mean residual: {np.mean(residuals):.4f}")  # Should be ~0
    print(f"Std residual:  {np.std(residuals):.4f}")
    
    # Plot residuals
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted')
    
    # Histogram of residuals
    axes[1].hist(residuals, bins=30, edgecolor='black')
    axes[1].set_xlabel('Residuals')
    axes[1].set_title('Residual Distribution')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, plot=axes[2])
    axes[2].set_title('Q-Q Plot')
    
    plt.tight_layout()
    plt.show()
```

---

## 3. Cross-Validation

```python
from sklearn.model_selection import (
    cross_val_score, cross_validate, KFold, StratifiedKFold,
    LeaveOneOut, TimeSeriesSplit, RepeatedKFold, GroupKFold
)

"""
Cross-validation: Estimate model performance on unseen data.

Why use CV:
- More reliable than single train/test split
- Uses all data for both training and validation
- Reduces variance of performance estimate
"""

# Basic K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")


# Stratified K-Fold (maintains class distribution in each fold)
# Use for imbalanced classification
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')


# Get multiple metrics
results = cross_validate(
    model, X, y, cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    return_train_score=True
)

print(f"Test Accuracy: {results['test_accuracy'].mean():.3f}")
print(f"Train Accuracy: {results['train_accuracy'].mean():.3f}")


# Leave-One-Out (LOO)
# Most expensive, useful for small datasets
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)


# Repeated K-Fold
repeated_kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(model, X, y, cv=repeated_kfold)


# Time Series Split (for temporal data)
# Preserves temporal order
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    print(f"Train: {train_idx[0]}-{train_idx[-1]}, Test: {test_idx[0]}-{test_idx[-1]}")


# Group K-Fold (groups must not overlap train/test)
# E.g., patients in medical data
groups = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]  # Group labels
group_kfold = GroupKFold(n_splits=4)

for train_idx, test_idx in group_kfold.split(X, y, groups):
    print(f"Train groups: {np.unique(np.array(groups)[train_idx])}")
    print(f"Test groups: {np.unique(np.array(groups)[test_idx])}")


# Manual cross-validation loop
def manual_cross_validation(model, X, y, cv=5):
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Clone model to avoid data leakage
        from sklearn.base import clone
        model_clone = clone(model)
        
        model_clone.fit(X_train, y_train)
        score = model_clone.score(X_val, y_val)
        scores.append(score)
        
        print(f"Fold {fold + 1}: {score:.4f}")
    
    return np.array(scores)
```

### Nested Cross-Validation

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

"""
Nested CV: Unbiased estimate when doing hyperparameter tuning.

Structure:
- Outer loop: Evaluate model performance
- Inner loop: Tune hyperparameters

Without nested CV, performance estimate is optimistically biased.
"""

def nested_cross_validation(model, param_grid, X, y, outer_cv=5, inner_cv=3):
    outer_scores = []
    
    outer_kfold = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
    
    for train_idx, test_idx in outer_kfold.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inner CV for hyperparameter tuning
        grid_search = GridSearchCV(
            model, param_grid, cv=inner_cv, scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)
        
        # Evaluate best model on outer test set
        score = grid_search.score(X_test, y_test)
        outer_scores.append(score)
        
        print(f"Best params: {grid_search.best_params_}, Score: {score:.4f}")
    
    return np.array(outer_scores)


# Using sklearn's cross_val_score with GridSearchCV
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='accuracy')
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='accuracy')

print(f"Nested CV Score: {nested_scores.mean():.3f} (+/- {nested_scores.std() * 2:.3f})")
```

---

## 4. Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

"""
Grid Search: Exhaustive search over parameter combinations.

Pros:
- Guaranteed to find best combination in grid
- Parallelizable

Cons:
- Exponential growth with parameters
- May miss optimal values between grid points
"""

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    refit=True  # Refit best model on full data
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)

# All results
results_df = pd.DataFrame(grid_search.cv_results_)
```

### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform

"""
Random Search: Sample random combinations from distributions.

Pros:
- More efficient than grid search
- Better for high-dimensional parameter spaces
- Can use continuous distributions

Cons:
- May miss optimal combination
- Results not reproducible without random_state
"""

param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9),
    'learning_rate': loguniform(1e-4, 1e-1)  # Log-uniform for learning rates
}

random_search = RandomizedSearchCV(
    model,
    param_distributions,
    n_iter=100,  # Number of random combinations to try
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")
```

### Bayesian Optimization

```python
"""
Bayesian Optimization: Build probabilistic model of objective function.

Pros:
- More efficient than random/grid search
- Works well with expensive evaluations
- Handles continuous parameters naturally

Libraries: Optuna, Hyperopt, Scikit-Optimize
"""

# Using Optuna
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
    }
    
    model = RandomForestClassifier(**params, random_state=42)
    
    # Cross-validation score
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return score.mean()

# Create study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value:.4f}")

# Get best model
best_params = study.best_params
best_model = RandomForestClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)


# Using Hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

space = {
    'n_estimators': hp.quniform('n_estimators', 100, 500, 50),
    'max_depth': hp.quniform('max_depth', 3, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
}

def hyperopt_objective(params):
    params = {k: int(v) for k, v in params.items()}
    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    return {'loss': -score, 'status': STATUS_OK}

trials = Trials()
best = fmin(
    fn=hyperopt_objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)
```

### Halving Grid/Random Search

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

"""
Successive Halving: Start with many configs on small data,
progressively increase data while eliminating poor performers.

Much faster than standard grid/random search.
"""

halving_search = HalvingGridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='accuracy',
    factor=3,  # Proportion of candidates selected each round
    resource='n_samples',  # or 'n_estimators' for iterative models
    min_resources='smallest',
    aggressive_elimination=False
)

halving_search.fit(X_train, y_train)
print(f"Best parameters: {halving_search.best_params_}")
```

---

## 5. Bias-Variance Tradeoff

```python
"""
Total Error = Bias² + Variance + Irreducible Error

Bias: Error from erroneous assumptions (underfitting)
- Model too simple
- High training error AND high test error
- Solutions: More features, complex model, less regularization

Variance: Error from sensitivity to fluctuations (overfitting)
- Model too complex
- Low training error BUT high test error
- Solutions: More data, simpler model, regularization, dropout

Irreducible Error: Noise in the data (cannot be reduced)
"""

import numpy as np
import matplotlib.pyplot as plt

def bias_variance_demo():
    # Generate data
    np.random.seed(42)
    X = np.linspace(0, 1, 100)
    y_true = np.sin(2 * np.pi * X)
    y = y_true + np.random.normal(0, 0.3, 100)
    
    # Fit models of different complexity
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    
    degrees = [1, 3, 15]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, degree in zip(axes, degrees):
        model = make_pipeline(
            PolynomialFeatures(degree),
            LinearRegression()
        )
        model.fit(X.reshape(-1, 1), y)
        y_pred = model.predict(X.reshape(-1, 1))
        
        ax.scatter(X, y, alpha=0.5, label='Data')
        ax.plot(X, y_true, 'g-', label='True function')
        ax.plot(X, y_pred, 'r-', label=f'Degree {degree}')
        ax.set_title(f'Polynomial Degree {degree}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()


# Learning Curves
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y, cv=5):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, 
                     train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, val_mean, 'o-', label='Validation score')
    plt.fill_between(train_sizes, val_mean - val_std, 
                     val_mean + val_std, alpha=0.1)
    
    plt.xlabel('Training set size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.show()
    
    """
    Interpretation:
    - High bias: Both curves converge to low score
    - High variance: Large gap between training and validation
    - Good fit: Both converge to high score with small gap
    """


# Validation Curves (hyperparameter effect)
from sklearn.model_selection import validation_curve

def plot_validation_curve(model, X, y, param_name, param_range, cv=5):
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring='accuracy', n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(param_range, train_mean, 'o-', label='Training score')
    plt.fill_between(param_range, train_mean - train_std, 
                     train_mean + train_std, alpha=0.1)
    plt.semilogx(param_range, val_mean, 'o-', label='Validation score')
    plt.fill_between(param_range, val_mean - val_std, 
                     val_mean + val_std, alpha=0.1)
    
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title('Validation Curve')
    plt.legend(loc='best')
    plt.show()
```

---

## 6. Interview Questions

```python
# Q1: When to use precision vs recall?
"""
High Precision priority:
- Spam detection (don't want to miss legitimate emails)
- Recommendation systems (user satisfaction)
- Cost of false positive is high

High Recall priority:
- Disease detection (don't want to miss sick patients)
- Fraud detection (don't want to miss fraud)
- Cost of false negative is high

Use F1 when both matter equally.
"""


# Q2: Why is accuracy misleading for imbalanced data?
"""
Example: 99% negative, 1% positive
- Predicting all negative gives 99% accuracy!
- But we miss all positives (0% recall)

Better metrics for imbalanced data:
- Precision, Recall, F1
- PR-AUC (preferred over ROC-AUC)
- Balanced accuracy
"""


# Q3: ROC-AUC vs PR-AUC?
"""
ROC-AUC:
- Plots TPR vs FPR
- Good for balanced datasets
- Can be misleading when negatives dominate

PR-AUC:
- Plots Precision vs Recall
- Better for imbalanced datasets
- Focuses on positive class performance
"""


# Q4: Why use cross-validation instead of single split?
"""
1. More reliable estimate of performance
2. Uses all data for both training and validation
3. Reduces variance of estimate
4. Detects overfitting better
5. Better use of limited data
"""


# Q5: What is data leakage and how to prevent it?
"""
Data leakage: Information from test set influences training.

Causes:
- Feature scaling before split
- Feature selection on full dataset
- Target encoding before split
- Time-series data not split chronologically

Prevention:
- Always split first, then preprocess
- Use pipelines for preprocessing
- Be careful with feature engineering
- Use proper time-series splitting
"""


# Q6: Grid search vs Random search?
"""
Grid Search:
- Exhaustive, guaranteed to find best in grid
- O(n^k) combinations
- Better for few hyperparameters

Random Search:
- More efficient for high dimensions
- Can explore continuous values
- Often finds good solutions faster
- Theoretically proven more efficient
"""


# Q7: How to detect overfitting?
"""
Signs of overfitting:
1. Training accuracy >> Validation accuracy
2. Complex model performs worse than simple
3. Performance degrades on new data

Detection methods:
1. Learning curves (gap between train/val)
2. Validation curves (score vs complexity)
3. Cross-validation
4. Hold-out test set

Solutions:
- More data
- Regularization
- Simpler model
- Early stopping
- Dropout (neural networks)
"""


# Q8: What is stratified sampling and when to use it?
"""
Stratified sampling: Maintain class proportions in each fold.

When to use:
- Imbalanced classification
- Small datasets
- Multi-class problems

Ensures each fold is representative of overall distribution.
"""


# Q9: Explain the difference between R² and adjusted R²
"""
R²: Proportion of variance explained
- Always increases when adding features (even useless ones)
- Can be misleading for model selection

Adjusted R²: Penalizes additional features
- R²_adj = 1 - (1-R²)(n-1)/(n-p-1)
- Only increases if new feature improves model more than by chance
- Better for comparing models with different feature counts
"""


# Q10: When would you use nested cross-validation?
"""
Use nested CV when:
- Doing hyperparameter tuning
- Want unbiased performance estimate
- Comparing different algorithms

Structure:
- Outer loop: Performance estimation
- Inner loop: Hyperparameter selection

Without nesting, performance estimate is optimistically biased
because test data influences hyperparameter selection.
"""
```

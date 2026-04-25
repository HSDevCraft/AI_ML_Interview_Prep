# Feature Engineering & Data Preprocessing - Complete Guide

## ⚡ Interview Quick Summary

> **Core insight**: Feature engineering is often more impactful than model selection. Good features encode domain knowledge and make patterns easier for the model to learn. Bad features introduce noise, leakage, or multicollinearity.

### Encoding Strategies for Categorical Variables

| Encoding | When to use | Pitfall |
|----------|-------------|--------|
| One-Hot | Low cardinality (< 20), linear models | Curse of dim for high cardinality |
| Label Encoding | Ordinal categories (S < M < L) | Don't use for nominal! |
| Target Encoding | High cardinality, gradient boosting | Data leakage — must use out-of-fold! |
| Frequency Encoding | High cardinality, when frequency signals quality | Loses category identity |
| Embedding | Very high cardinality (user IDs) | Requires retraining for new values |
| Binary Encoding | High cardinality, need compression | Less interpretable |

### Handling Missing Values — Strategy Decision Tree

```
Is missing random (MCAR) or informative (MNAR)?

  MCAR (truly random):
    Numerical:    mean/median imputation, KNN imputation
    Categorical:  mode imputation, 'Unknown' category
    Many missing: consider dropping feature if >50% missing

  MNAR (not random — missingness carries signal):
    Add a binary 'is_missing' indicator feature FIRST
    Then impute the original feature
    e.g., loan_amount missing might mean loan was rejected

  Time series: forward fill (last known value), interpolation
```

### Feature Scaling — When It Matters

```
Needs scaling:         Distance/gradient sensitive
  StandardScaler      Zero mean, unit variance (most common)
  MinMaxScaler        Range [0,1] (for neural nets, image pixels)
  RobustScaler        Robust to outliers (uses IQR)

Doesn't need scaling:  Tree-based models
  Decision Trees       Split on rank, not value
  Random Forest        Same
  XGBoost/LightGBM     Same

CRITICAL: Fit scaler on TRAIN data, transform TRAIN and TEST
  Bad:  scaler.fit_transform(all_data)  ← leaks test distribution!
  Good: scaler.fit(X_train); scaler.transform(X_test)
```

### Feature Selection Methods

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.inspection import permutation_importance

# Filter method: statistical test (fast, model-agnostic)
sel = SelectKBest(f_classif, k=20)
X_selected = sel.fit_transform(X_train, y_train)

# Wrapper: Recursive Feature Elimination (model-specific, expensive)
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=20)
rfe.fit(X_train, y_train)

# Embedded: permutation importance (unbiased, recommended)
perm = permutation_importance(model, X_val, y_val, n_repeats=10)
important = perm.importances_mean > 0.001  # threshold
```

### 🚨 Top Interview Pitfalls
- **Target leakage**: using features that are derived from or contain the target (e.g., using total charges to predict churn when total charges depend on whether they churned)
- **Train-test leakage**: fitting scaler/imputer/encoder on full dataset before split
- Using **label encoding for nominal categories** (SVM, neural nets assume ordinal relationship in numbers)
- **Target encoding without out-of-fold**: causes massive overfitting; always compute target means on other folds

---

## Table of Contents
1. [Data Preprocessing](#data-preprocessing)
2. [Feature Scaling](#feature-scaling)
3. [Encoding Categorical Variables](#encoding-categorical-variables)
4. [Feature Selection](#feature-selection)
5. [Feature Creation](#feature-creation)
6. [Handling Imbalanced Data](#handling-imbalanced-data)
7. [Interview Questions](#interview-questions)

---

## 1. Data Preprocessing

### Handling Missing Values

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# Detect missing values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': ['x', 'y', None, 'z']
})

# Missing value analysis
print(df.isnull().sum())  # Count per column
print(df.isnull().sum() / len(df) * 100)  # Percentage

# Visualize missing patterns
import missingno as msno
msno.matrix(df)
msno.heatmap(df)


# Strategy 1: Drop rows/columns
df_dropped_rows = df.dropna()  # Drop rows with any NaN
df_dropped_cols = df.dropna(axis=1)  # Drop columns with any NaN
df_thresh = df.dropna(thresh=2)  # Keep rows with at least 2 non-NaN


# Strategy 2: Simple Imputation
imputer_mean = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')
imputer_mode = SimpleImputer(strategy='most_frequent')
imputer_const = SimpleImputer(strategy='constant', fill_value=0)

df_numeric = df[['A', 'B']]
df_imputed = pd.DataFrame(
    imputer_mean.fit_transform(df_numeric),
    columns=df_numeric.columns
)


# Strategy 3: KNN Imputation
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
df_knn_imputed = pd.DataFrame(
    knn_imputer.fit_transform(df_numeric),
    columns=df_numeric.columns
)


# Strategy 4: Iterative Imputation (MICE)
from sklearn.experimental import enable_iterative_imputer
iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
df_iter_imputed = pd.DataFrame(
    iterative_imputer.fit_transform(df_numeric),
    columns=df_numeric.columns
)


# Strategy 5: Indicator variable for missingness
df['A_missing'] = df['A'].isnull().astype(int)
df['A'] = df['A'].fillna(df['A'].median())


# Strategy 6: Group-based imputation
df['A'] = df.groupby('C')['A'].transform(lambda x: x.fillna(x.mean()))


# Best practices
"""
1. Understand WHY data is missing (MCAR, MAR, MNAR)
2. Consider domain knowledge
3. Don't impute test set with test set statistics
4. Use pipelines to avoid data leakage
5. Consider missingness as a feature
"""
```

### Handling Outliers

```python
import numpy as np
import pandas as pd
from scipy import stats

# Detection methods

# Method 1: Z-Score
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold

# Method 2: IQR (Interquartile Range)
def detect_outliers_iqr(data, multiplier=1.5):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return (data < lower) | (data > upper)

# Method 3: Modified Z-Score (more robust)
def detect_outliers_modified_zscore(data, threshold=3.5):
    median = np.median(data)
    mad = np.median(np.abs(data - median))  # Median Absolute Deviation
    modified_z = 0.6745 * (data - median) / mad
    return np.abs(modified_z) > threshold

# Method 4: Isolation Forest
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outlier_pred = iso_forest.fit_predict(X)  # -1 for outliers


# Treatment methods

# Method 1: Remove outliers
df_clean = df[~detect_outliers_iqr(df['value'])]

# Method 2: Cap/Clip (Winsorization)
lower = df['value'].quantile(0.01)
upper = df['value'].quantile(0.99)
df['value_capped'] = df['value'].clip(lower, upper)

# Method 3: Transform (log, sqrt)
df['value_log'] = np.log1p(df['value'])  # log(1+x) handles zeros

# Method 4: Binning
df['value_binned'] = pd.cut(df['value'], bins=10)

# Method 5: Robust scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()  # Uses median and IQR
df_scaled = scaler.fit_transform(df[['value']])


# Best practices
"""
1. Investigate outliers before removing (may be valuable!)
2. Domain knowledge is crucial
3. Document outlier treatment
4. Consider robust algorithms (e.g., Huber loss)
"""
```

### Data Cleaning

```python
import pandas as pd

# Duplicate handling
df.duplicated().sum()  # Count duplicates
df_deduped = df.drop_duplicates()
df_deduped = df.drop_duplicates(subset=['id'])  # Based on specific columns
df_deduped = df.drop_duplicates(keep='last')  # Keep last occurrence

# Data type conversion
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')
df['id'] = df['id'].astype(str)
df['value'] = pd.to_numeric(df['value'], errors='coerce')  # Invalid to NaN

# String cleaning
df['text'] = df['text'].str.strip()  # Remove whitespace
df['text'] = df['text'].str.lower()  # Lowercase
df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)  # Remove punctuation

# Inconsistent categories
df['category'] = df['category'].str.lower().str.strip()
df['category'] = df['category'].replace({
    'yes': 'Yes', 'YES': 'Yes',
    'no': 'No', 'NO': 'No'
})

# Invalid values
df.loc[df['age'] < 0, 'age'] = np.nan
df.loc[df['age'] > 120, 'age'] = np.nan
```

---

## 2. Feature Scaling

```python
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, Normalizer, PowerTransformer, QuantileTransformer
)

X = np.array([[1, 2, 100], [2, 3, 200], [3, 4, 300], [4, 5, 400]])

# StandardScaler (Z-score normalization)
# x_scaled = (x - mean) / std
# Result: mean=0, std=1
standard = StandardScaler()
X_standard = standard.fit_transform(X)

# MinMaxScaler (normalize to range [0, 1])
# x_scaled = (x - min) / (max - min)
minmax = MinMaxScaler(feature_range=(0, 1))
X_minmax = minmax.fit_transform(X)

# RobustScaler (uses median and IQR - robust to outliers)
# x_scaled = (x - median) / IQR
robust = RobustScaler()
X_robust = robust.fit_transform(X)

# MaxAbsScaler (scales to [-1, 1] by max absolute value)
# Preserves sparsity (doesn't shift data)
maxabs = MaxAbsScaler()
X_maxabs = maxabs.fit_transform(X)

# Normalizer (scales each sample to unit norm)
# Different from other scalers - normalizes rows, not columns
normalizer = Normalizer(norm='l2')  # 'l1', 'l2', 'max'
X_normalized = normalizer.fit_transform(X)

# PowerTransformer (make data more Gaussian-like)
# Box-Cox (requires positive data) or Yeo-Johnson
power = PowerTransformer(method='yeo-johnson')
X_power = power.fit_transform(X)

# QuantileTransformer (uniform or normal output distribution)
quantile = QuantileTransformer(output_distribution='normal')
X_quantile = quantile.fit_transform(X)


# When to use which scaler
"""
StandardScaler:
- Default choice for many algorithms
- When features are roughly Gaussian
- SVM, Logistic Regression, Neural Networks

MinMaxScaler:
- When you need bounded range [0, 1]
- Neural Networks, Image data
- Not good with outliers

RobustScaler:
- When data has outliers
- Based on percentiles, not affected by extremes

PowerTransformer:
- When features are highly skewed
- Makes data more Gaussian

QuantileTransformer:
- When you want uniform/normal distribution
- Spreads out clusters

No scaling needed:
- Tree-based models (RF, XGBoost, etc.)
- Naive Bayes
"""


# Important: Always fit on training data only!
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit AND transform
X_test_scaled = scaler.transform(X_test)  # Only transform!
```

---

## 3. Encoding Categorical Variables

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    LabelEncoder, OrdinalEncoder, OneHotEncoder
)

# Sample data
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['S', 'M', 'L', 'XL', 'M'],
    'brand': ['A', 'B', 'A', 'C', 'B']
})


# Label Encoding (for ordinal features or target)
# Maps categories to integers 0, 1, 2, ...
le = LabelEncoder()
df['color_encoded'] = le.fit_transform(df['color'])
# Can inverse: le.inverse_transform([0, 1, 2])


# Ordinal Encoding (when order matters)
# Specify the order of categories
ordinal_encoder = OrdinalEncoder(categories=[['S', 'M', 'L', 'XL']])
df['size_encoded'] = ordinal_encoder.fit_transform(df[['size']])


# One-Hot Encoding (for nominal features)
# Creates binary columns for each category
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = ohe.fit_transform(df[['color']])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out())

# Pandas get_dummies (simpler alternative)
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True)


# Target Encoding (Mean Encoding)
# Replace category with mean of target variable
def target_encode(df, column, target, smoothing=1):
    """
    Encode categorical with target mean (with smoothing).
    Smoothing helps with rare categories.
    """
    global_mean = df[target].mean()
    agg = df.groupby(column)[target].agg(['mean', 'count'])
    
    # Smoothed mean
    smooth = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
    
    return df[column].map(smooth)

# Use with cross-validation to avoid leakage!
from sklearn.model_selection import KFold

def target_encode_cv(df, column, target, n_splits=5):
    """Target encoding with cross-validation to prevent leakage."""
    df = df.copy()
    df['encoded'] = np.nan
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kfold.split(df):
        means = df.iloc[train_idx].groupby(column)[target].mean()
        df.loc[df.index[val_idx], 'encoded'] = df.iloc[val_idx][column].map(means)
    
    # Fill NaN with global mean
    df['encoded'] = df['encoded'].fillna(df[target].mean())
    
    return df['encoded']


# Frequency Encoding
# Replace category with its frequency
freq_map = df['color'].value_counts(normalize=True).to_dict()
df['color_freq'] = df['color'].map(freq_map)


# Binary Encoding
# Encode as binary representation (fewer columns than one-hot)
import category_encoders as ce  # pip install category_encoders

binary_encoder = ce.BinaryEncoder(cols=['color'])
df_binary = binary_encoder.fit_transform(df)


# Hash Encoding (for high-cardinality features)
# Fixed number of output columns
hash_encoder = ce.HashingEncoder(cols=['brand'], n_components=8)
df_hashed = hash_encoder.fit_transform(df)


# Leave-One-Out Encoding
loo_encoder = ce.LeaveOneOutEncoder(cols=['color'])
df_loo = loo_encoder.fit_transform(df, df['target'])


# When to use which encoding
"""
Label/Ordinal Encoding:
- Ordinal features (size: S, M, L, XL)
- Tree-based models can handle arbitrary integers

One-Hot Encoding:
- Nominal features (color: red, blue, green)
- Low cardinality (< 10-15 categories)
- Linear models

Target Encoding:
- High cardinality features
- Use CV to avoid leakage
- Tree-based and linear models

Binary/Hash Encoding:
- Very high cardinality (1000s of categories)
- Memory efficient
"""
```

---

## 4. Feature Selection

```python
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, chi2,
    mutual_info_classif, mutual_info_regression,
    RFE, RFECV, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier

# Sample data
X, y = np.random.randn(100, 20), np.random.randint(0, 2, 100)


# Filter Methods (statistical tests)

# Variance Threshold (remove low-variance features)
selector = VarianceThreshold(threshold=0.1)
X_var = selector.fit_transform(X)
print(f"Features kept: {selector.get_support().sum()}")

# SelectKBest with statistical tests
# Classification: f_classif, chi2, mutual_info_classif
# Regression: f_regression, mutual_info_regression

selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Get scores and selected features
scores = selector.scores_
selected_features = selector.get_support(indices=True)

# Mutual Information (captures non-linear relationships)
mi_selector = SelectKBest(score_func=mutual_info_classif, k=10)
X_mi = mi_selector.fit_transform(X, y)


# Wrapper Methods (use model performance)

# Recursive Feature Elimination (RFE)
rfe = RFE(
    estimator=RandomForestClassifier(n_estimators=100),
    n_features_to_select=10,
    step=1  # Number of features to remove per iteration
)
X_rfe = rfe.fit_transform(X, y)
print(f"Selected features: {rfe.support_}")
print(f"Feature ranking: {rfe.ranking_}")

# RFE with Cross-Validation
rfecv = RFECV(
    estimator=RandomForestClassifier(n_estimators=100),
    step=1,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
rfecv.fit(X, y)
print(f"Optimal features: {rfecv.n_features_}")


# Embedded Methods (feature importance from model)

# SelectFromModel
selector = SelectFromModel(
    estimator=RandomForestClassifier(n_estimators=100),
    threshold='median'  # or '0.5*mean', 0.1, etc.
)
X_selected = selector.fit_transform(X, y)

# Get feature importances
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
importances = rf.feature_importances_

# L1 regularization (Lasso) for selection
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5)
lasso.fit(X, y)
selected = np.abs(lasso.coef_) > 0


# Correlation-based selection
def remove_correlated_features(df, threshold=0.9):
    """Remove highly correlated features."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop)


# Permutation Importance (model-agnostic)
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(
    rf, X, y, n_repeats=10, random_state=42, n_jobs=-1
)
sorted_idx = perm_importance.importances_mean.argsort()[::-1]


# Best practices
"""
1. Start with filter methods (fast)
2. Use wrapper methods for refinement
3. Consider domain knowledge
4. Use CV to validate selection
5. Be careful of data leakage (fit on train only!)
"""
```

---

## 5. Feature Creation

```python
import pandas as pd
import numpy as np

# Datetime features
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['hour'] = df['date'].dt.hour
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

# Cyclical encoding for periodic features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)


# Mathematical transformations
df['log_value'] = np.log1p(df['value'])  # Log transform
df['sqrt_value'] = np.sqrt(df['value'])
df['square_value'] = df['value'] ** 2
df['reciprocal'] = 1 / (df['value'] + 1)


# Interaction features
df['A_times_B'] = df['A'] * df['B']
df['A_div_B'] = df['A'] / (df['B'] + 1)
df['A_plus_B'] = df['A'] + df['B']
df['A_minus_B'] = df['A'] - df['B']

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X)


# Aggregation features (for grouped data)
df['mean_by_category'] = df.groupby('category')['value'].transform('mean')
df['std_by_category'] = df.groupby('category')['value'].transform('std')
df['count_by_category'] = df.groupby('category')['value'].transform('count')
df['max_by_category'] = df.groupby('category')['value'].transform('max')
df['min_by_category'] = df.groupby('category')['value'].transform('min')
df['value_vs_category_mean'] = df['value'] - df['mean_by_category']


# Binning/Discretization
df['age_binned'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100],
                          labels=['child', 'young', 'middle', 'senior', 'elderly'])

# Quantile binning
df['value_quantile'] = pd.qcut(df['value'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])


# Text features
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['capital_count'] = df['text'].str.count(r'[A-Z]')
df['digit_count'] = df['text'].str.count(r'\d')


# Lag features (time series)
df['value_lag_1'] = df['value'].shift(1)
df['value_lag_7'] = df['value'].shift(7)
df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
df['rolling_std_7'] = df['value'].rolling(window=7).std()
df['ewm_mean'] = df['value'].ewm(span=7).mean()


# Ratio features
df['price_per_sqft'] = df['price'] / df['sqft']
df['completion_rate'] = df['completed'] / df['total']


# Domain-specific examples

# E-commerce
df['cart_value'] = df['price'] * df['quantity']
df['discount_pct'] = df['discount'] / df['original_price']

# Finance
df['debt_to_income'] = df['debt'] / df['income']
df['credit_utilization'] = df['balance'] / df['credit_limit']

# Healthcare
df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
```

---

## 6. Handling Imbalanced Data

```python
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN

# pip install imbalanced-learn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


# Oversampling Methods (increase minority class)

# Random Oversampling (duplicate minority samples)
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_train, y_train)

# SMOTE (Synthetic Minority Over-sampling Technique)
# Creates synthetic samples by interpolating between neighbors
smote = SMOTE(random_state=42, k_neighbors=5)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

# ADASYN (Adaptive Synthetic Sampling)
# Generates more samples in difficult regions
adasyn = ADASYN(random_state=42)
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)

# BorderlineSMOTE (focus on borderline samples)
from imblearn.over_sampling import BorderlineSMOTE
bsmote = BorderlineSMOTE(random_state=42)
X_bsmote, y_bsmote = bsmote.fit_resample(X_train, y_train)


# Undersampling Methods (decrease majority class)

# Random Undersampling
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train, y_train)

# Tomek Links (remove ambiguous pairs)
tomek = TomekLinks()
X_tomek, y_tomek = tomek.fit_resample(X_train, y_train)

# NearMiss (select majority samples closest to minority)
nearmiss = NearMiss(version=1)
X_nm, y_nm = nearmiss.fit_resample(X_train, y_train)


# Combination Methods

# SMOTE + Tomek Links
smote_tomek = SMOTETomek(random_state=42)
X_st, y_st = smote_tomek.fit_resample(X_train, y_train)

# SMOTE + ENN (Edited Nearest Neighbors)
smote_enn = SMOTEENN(random_state=42)
X_se, y_se = smote_enn.fit_resample(X_train, y_train)


# Algorithm-Level Solutions

# Class weights
from sklearn.ensemble import RandomForestClassifier

# Auto-compute weights inversely proportional to class frequencies
rf_weighted = RandomForestClassifier(class_weight='balanced', random_state=42)

# Manual class weights
rf_weighted = RandomForestClassifier(
    class_weight={0: 1, 1: 10},  # Penalize misclassifying class 1 more
    random_state=42
)


# Cost-sensitive learning
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(class_weight='balanced')


# Threshold adjustment
from sklearn.metrics import precision_recall_curve

y_proba = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# Find threshold for desired precision or recall
def find_threshold(precision_target, precisions, recalls, thresholds):
    idx = np.argmin(np.abs(precisions[:-1] - precision_target))
    return thresholds[idx]


# Ensemble methods for imbalanced data
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

# Balanced Random Forest
brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
brf.fit(X_train, y_train)

# Easy Ensemble (AdaBoost on balanced subsets)
ee = EasyEnsembleClassifier(n_estimators=10, random_state=42)
ee.fit(X_train, y_train)


# Best practices
"""
1. Never resample test set! Only training data.
2. Apply resampling AFTER train/test split
3. Use stratified splits to maintain proportions
4. Prefer PR-AUC over ROC-AUC for evaluation
5. Consider cost of different errors
6. Combine multiple approaches
7. Don't blindly balance - consider domain requirements
"""
```

---

## 7. Interview Questions

```python
# Q1: When should you NOT impute missing values?
"""
1. Missingness is informative (create indicator instead)
2. Very high percentage missing (>50%)
3. Domain knowledge says value should be NA
4. Using algorithms that handle missing values (XGBoost, LightGBM)
"""


# Q2: What's the difference between standardization and normalization?
"""
Standardization (Z-score):
- x_new = (x - mean) / std
- No bounded range
- Assumes roughly Gaussian distribution
- Less sensitive to outliers

Normalization (Min-Max):
- x_new = (x - min) / (max - min)
- Bounded to [0, 1]
- Sensitive to outliers
- Good for bounded algorithms (neural networks)
"""


# Q3: Why might one-hot encoding be problematic?
"""
1. High cardinality → many sparse columns
2. Curse of dimensionality
3. Memory issues with large datasets
4. Unseen categories at test time
5. Tree models don't benefit much

Alternatives:
- Target encoding (with CV)
- Embedding layers (deep learning)
- Hash encoding
- Group rare categories
"""


# Q4: What is target leakage and how to prevent it?
"""
Target leakage: Using information that wouldn't be available at prediction time.

Examples:
- Feature derived from target
- Future information in time series
- Data that's collected after the target event

Prevention:
- Understand data collection process
- Time-based validation for temporal data
- Use pipelines with proper fit/transform
- Feature selection on training data only
"""


# Q5: When to use SMOTE vs class weights?
"""
SMOTE (oversampling):
- Creates synthetic samples
- Increases dataset size
- May create noisy samples
- Good when minority class is small

Class Weights:
- No new samples created
- Adjusts loss function
- More computationally efficient
- Better for moderate imbalance

Combined approach often works best.
"""


# Q6: How to handle categorical features with many levels?
"""
Options:
1. Target encoding (with cross-validation)
2. Frequency encoding
3. Embedding layers (neural networks)
4. Hash encoding
5. Group rare categories into 'Other'
6. Keep top N categories, bin rest
7. Domain-specific groupings
"""


# Q7: What's the difference between filter, wrapper, and embedded feature selection?
"""
Filter methods:
- Use statistical tests (correlation, chi-square)
- Fast, model-independent
- May miss feature interactions

Wrapper methods:
- Use model performance (RFE, forward/backward)
- Find best subset for specific model
- Computationally expensive

Embedded methods:
- Selection during model training
- L1 regularization, tree importance
- Balance of speed and accuracy
"""


# Q8: Why is feature scaling important for some algorithms?
"""
Algorithms requiring scaling:
- Distance-based: KNN, K-Means, SVM
- Gradient-based: Neural Networks, Logistic Regression
- Regularized: Ridge, Lasso, ElasticNet

Why:
- Different scales dominate distance calculations
- Gradient magnitudes differ
- Regularization penalizes features unequally

Not required:
- Tree-based models (splits based on thresholds)
- Naive Bayes (probability-based)
"""


# Q9: How to prevent data leakage in cross-validation?
"""
1. Use pipelines for all preprocessing
2. Fit preprocessors only on training folds
3. Feature engineering before split OR inside CV
4. Time-based splits for temporal data
5. Group splits for correlated samples
6. Never use test set statistics
"""

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Pipeline ensures fit only on training data in each fold
scores = cross_val_score(pipeline, X, y, cv=5)


# Q10: What features would you create for a time series problem?
"""
Lag features:
- value_t-1, value_t-7, value_t-30

Rolling statistics:
- rolling_mean, rolling_std, rolling_min, rolling_max

Time features:
- hour, day_of_week, month, quarter, year
- is_weekend, is_holiday

Cyclical encoding:
- sin/cos transforms for periodic features

Differences:
- value_t - value_t-1, percent_change

Domain-specific:
- moving averages, trend indicators
"""
```

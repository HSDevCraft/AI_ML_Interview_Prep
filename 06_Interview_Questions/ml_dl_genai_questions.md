# ML / DL / GenAI Interview Questions Bank

> **How to use**: Practice the **one-liner answer first** (recall), then build depth. Always link theory to a real project. Mention pitfalls proactively — interviewers reward self-awareness.

## Interview Strategy

| Round | Focus | Key Signal |
|-------|-------|------------|
| ML Theory | Bias-variance, regularization, metrics | Can you derive formulas and explain trade-offs? |
| DL Theory | Backprop, attention, normalization | Can you explain WHY, not just WHAT? |
| GenAI/LLM | Transformers, fine-tuning, RAG, agents | Can you discuss production trade-offs? |
| System Design | End-to-end ML system | Trade-offs, scale, failure modes |

### PEDAL Framework for Technical Answers
1. **P**roblem — define it precisely
2. **E**quation — write the math
3. **D**iagram — sketch or describe architecture
4. **A**nalogy — build intuition
5. **L**imitation — what breaks it, trade-offs

---

## Table of Contents
1. [Machine Learning Questions](#machine-learning-questions)
2. [Deep Learning Questions](#deep-learning-questions)
3. [GenAI & LLM Questions](#genai--llm-questions)
4. [MLOps Questions](#mlops-questions)
5. [System Design Questions](#system-design-questions)
6. [Behavioral Questions](#behavioral-questions)

---

## Machine Learning Questions

### Q1: Explain Bias-Variance Tradeoff

> **⚡ One-liner**: Bias = error from wrong assumptions (underfitting); Variance = error from sensitivity to training data (overfitting). Total Error = Bias² + Variance + Irreducible Noise.

**💡 Intuition**: A straight line (high bias) always misses a curved truth. A degree-100 polynomial (high variance) memorizes every noise point. The goal is the sweet spot.

```
Total Error = Bias² + Variance + Irreducible Error

                Train Error    Test Error    Fix
Underfitting       High           High     → Add complexity, more features
Overfitting        Low            High     → Regularize, more data, ensemble
Good Fit           Low            ~Low     ✓
```

```python
import numpy as np

# Visualize tradeoff: how complexity affects bias vs variance
X = np.linspace(0, 1, 100)
y_true  = np.sin(2 * np.pi * X)
y_noisy = y_true + np.random.normal(0, 0.3, len(X))

for deg in [1, 4, 15]:  # underfit, good fit, overfit
    coeffs  = np.polyfit(X, y_noisy, deg)
    y_pred  = np.polyval(coeffs, X)
    train_e = np.mean((y_noisy - y_pred) ** 2)
    test_e  = np.mean((y_true  - y_pred) ** 2)  # vs true signal
    print(f"deg={deg:2d}  train={train_e:.3f}  test={test_e:.3f}")
# deg= 1  train=0.180  test=0.500  ← underfitting (high bias)
# deg= 4  train=0.090  test=0.100  ← good fit
# deg=15  train=0.001  test=0.800  ← overfitting (high variance)
```

**🚨 Common Pitfalls**:
- Forgetting **irreducible error** (noise no model can remove)
- More data reduces variance but **NOT bias**
- Early stopping reduces variance; increasing model depth reduces bias

**💬 Follow-ups**: "How does ensemble learning reduce variance?" → Averaging multiple models cancels individual noise. "Can a model have both high bias and high variance?" → Yes: misspecified model on small dataset.

### Q2: What is regularization? Compare L1 vs L2

> **⚡ One-liner**: Regularization adds a penalty on weight magnitude to prevent overfitting. L1 (Lasso) zeroes out unimportant weights (feature selection); L2 (Ridge) shrinks all weights toward zero evenly.

**💡 Intuition for sparsity**: L1 ball has sharp corners on axes → gradient descent hits corners where some w=0. L2 ball is smooth → gradient descent stops before hitting an axis.

```
L1 (Lasso):    Loss + λ Σ|w|       → sparse weights, some exactly 0
L2 (Ridge):    Loss + λ Σw²        → all small, non-zero weights
Elastic Net:   Loss + λ₁ Σ|w| + λ₂ Σw²  → best of both

Gradient comparison:
  L1: sign(w) × λ      — constant push (same force regardless of weight size)
  L2: 2λw              — proportional push (large weights get pushed more)
  Why L2 never reaches 0: gradient at w=0 is 2λ×0=0, no force to cross zero
```

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso

X = np.random.randn(100, 10)  # 10 features, many irrelevant
y = X[:, 0] * 3 + X[:, 1] * 2 + np.random.randn(100) * 0.5  # only 2 matter

ridge = Ridge(alpha=1.0).fit(X, y)
lasso = Lasso(alpha=0.1).fit(X, y)

print("Ridge:", np.round(ridge.coef_, 2))
# → [2.9, 1.9, 0.1, -0.1, 0.2, ...]  all non-zero (shrunk)
print("Lasso:", np.round(lasso.coef_, 2))
# → [2.9, 1.9, 0.0,  0.0, 0.0, ...]  sparse (irrelevant → exactly 0)
```

**🚨 Common Pitfalls**:
- Forgetting to **standardize features** before regularization (otherwise penalty is unfair across scales)
- Confusing λ direction: larger λ = MORE regularization = simpler model
- L1 doesn't always zero out features — depends on λ strength

**💬 Follow-up**: "Why doesn't L2 give exact zeros?" → Gradient at zero is `2λ·0=0`, so no force pushes through zero.

### Q3: Explain precision, recall, F1-score. When to use each?

> **⚡ One-liner**: Precision = "of what I flagged positive, how many were?" Recall = "of all actual positives, how many did I catch?" F1 is their harmonic mean — use it when FP and FN costs both matter.

**💡 Net analogy**: Wide net = high recall (catch everything) but low precision (lots of bycatch). Narrow net = high precision but low recall.

```
Confusion Matrix:
                 Predicted +     Predicted -
Actual +      TP (caught)      FN (missed)
Actual -      FP (false alarm)  TN (correct reject)

Precision = TP / (TP + FP)  → minimize FP  (spam filter: don't block real mail)
Recall    = TP / (TP + FN)  → minimize FN  (cancer: don't miss cases)
F1        = 2·P·R / (P+R)   → harmonic mean, punishes extreme imbalance
PR-AUC    → use for highly imbalanced data (<5% positive class)
ROC-AUC   → comparing classifiers across all thresholds
```

```python
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import numpy as np

y_true   = [1, 1, 1, 0, 0, 0, 1, 0]
y_pred   = [1, 1, 0, 0, 1, 0, 1, 1]  # TP=3, FP=2, FN=1

print(f"Precision: {precision_score(y_true, y_pred):.2f}")  # 3/(3+2) = 0.60
print(f"Recall:    {recall_score(y_true, y_pred):.2f}")     # 3/(3+1) = 0.75
print(f"F1:        {f1_score(y_true, y_pred):.2f}")         # 0.667

# Threshold tuning: trade precision for recall
y_scores = np.array([0.9, 0.8, 0.4, 0.1, 0.7, 0.2, 0.85, 0.6])
precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
# Lower threshold → higher recall, lower precision (catch more, more false alarms)
```

**🚨 Common Pitfalls**:
- Using **accuracy on imbalanced data** — 99% accuracy predicting all-negative on 1% fraud data
- Not specifying macro/micro/weighted for multi-class F1
- Precision and recall are threshold-dependent — always report at your deployment threshold

**💬 Follow-up**: "Macro vs micro F1?" → Macro = average per-class F1; Micro = aggregate TP/FP/FN (better for class imbalance).

### Q4: How does Random Forest work? Why is it effective?

> **⚡ One-liner**: Random Forest trains many decorrelated trees on bootstrap samples with random feature subsets, then averages predictions — reducing variance while keeping bias low.

**💡 Intuition**: 100 diverse experts who each see different parts of the data will collectively be more reliable than one expert who sees everything — especially if they're not all making the same mistakes (decorrelation).

```
Algorithm:
  For each of B trees:
    1. Sample n rows WITH replacement (bootstrap) → ~63% unique samples
    2. At each split, consider only sqrt(p) random features
    3. Grow tree fully (high variance but low bias individually)
  Predict: majority vote (classification) or average (regression)

Why random features matter:
  Without them: all trees see same strong features → highly correlated
  With them:    trees disagree on different subspaces → variance cancels on averaging

OOB (Out-of-Bag) Score:
  ~37% of samples unused per tree → free validation without separate test set

Key hyperparameters:
  n_estimators:     More trees = better (diminishing returns after ~200)
  max_features:     sqrt(p) for classification, p/3 for regression
  max_depth:        None = grow fully (low bias), limit for regularization
  min_samples_leaf: Higher = more regularization, smoother boundaries
```

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np

rf = RandomForestClassifier(
    n_estimators=200,
    max_features='sqrt',    # only sqrt(p) features per split
    oob_score=True,         # free validation estimate
    n_jobs=-1,              # parallel training
    random_state=42
).fit(X_train, y_train)

print(f"OOB Score (free validation): {rf.oob_score_:.3f}")

# PITFALL: impurity-based importance is biased toward high-cardinality features
# Use permutation importance for unbiased estimates:
perm = permutation_importance(rf, X_val, y_val, n_repeats=10)
for i in np.argsort(perm.importances_mean)[::-1][:5]:
    print(f"Feature {i}: {perm.importances_mean[i]:.4f} ± {perm.importances_std[i]:.4f}")
```

**🚨 Common Pitfalls**:
- `feature_importances_` is **biased toward high-cardinality features** → use `permutation_importance`
- Forgetting `oob_score=True` — it's a free sanity check during training
- Very high `n_estimators` with no parallelism (`n_jobs=-1`) is slow with no benefit

**💬 Follow-ups**: "Bagging vs boosting?" → Bagging = parallel on bootstrap samples (reduces variance); boosting = sequential on residuals (reduces bias). "When does RF fail?" → Linear relationships, very sparse high-dim data.

### Q5: Explain gradient boosting. How does XGBoost improve on it?

> **⚡ One-liner**: Gradient boosting sequentially adds weak trees, each fitting the residuals of the previous ensemble. XGBoost improves it with L1/L2 regularization, second-order gradients, parallel split finding, and sparse data handling.

**💡 Intuition**: Think of correcting mistakes iteratively. If your first model predicts 70 when truth is 100, the next model learns to predict the error (30). You accumulate these corrections until the ensemble is accurate.

```
Gradient Boosting Algorithm:
  F_0(x) = mean(y)                   ← start with constant
  For t = 1 to T:
    r_t = y - F_{t-1}(x)             ← compute residuals (pseudo-residuals)
    h_t = fit tree to r_t             ← learn to predict errors
    F_t = F_{t-1} + η·h_t            ← add with learning rate η (shrinkage)

  More general: fit h_t to -∂L/∂F_{t-1}  (gradient of loss in function space)

XGBoost Key Improvements:
  1. Regularization: Ω(f) = γT + ½λ||w||²  (penalize # leaves T and leaf weights w)
  2. Second-order Taylor expansion: uses Hessian H for better approximation of loss
     Optimal leaf weight: w*_j = -G_j / (H_j + λ)  where G=sum grad, H=sum hessian
  3. Column (feature) subsampling: decorrelates trees (like RF)
  4. Sparse-aware split finding: efficient handling of missing values
  5. Block structure: cache-optimized for parallel computation
  6. Early stopping with eval_metric monitoring
```

```python
import xgboost as xgb

# XGBoost with all key parameters explained
model = xgb.XGBClassifier(
    n_estimators=500,       # max trees (use early stopping in practice)
    learning_rate=0.05,     # η: smaller = more trees needed, better generalization
    max_depth=6,            # tree depth (3-8 typical; deeper = more variance)
    subsample=0.8,          # row subsampling per tree (reduces overfitting)
    colsample_bytree=0.8,   # feature subsampling per tree (like RF)
    reg_alpha=0.1,          # L1 on leaf weights
    reg_lambda=1.0,         # L2 on leaf weights
    eval_metric='logloss',
    early_stopping_rounds=20,   # stop if no improvement for 20 rounds
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(f"Best iteration: {model.best_iteration}")
```

**🚨 Common Pitfalls**:
- Not using **early stopping** → trees keep adding, overfitting
- Too high `learning_rate` (>0.1) → model doesn't generalize well; lower lr + more trees is usually better
- Ignoring `scale_pos_weight` for imbalanced data: `scale_pos_weight = negative_count / positive_count`

**💬 Follow-ups**: "XGBoost vs LightGBM?" → LightGBM uses leaf-wise tree growth and histogram binning for speed; XGBoost uses level-wise growth. LightGBM is faster for large datasets. "XGBoost vs CatBoost?" → CatBoost handles categorical features natively with target encoding.

### Q6: What is cross-validation? Why use it?
```
Cross-validation: Technique to assess model generalization

K-Fold CV:
1. Split data into K folds
2. Train on K-1 folds, validate on 1
3. Repeat K times, each fold as validation once
4. Average metrics across folds

Benefits:
- Better estimate of model performance
- Uses all data for training and validation
- Reduces variance in performance estimate
- Helps detect overfitting

Variations:
- Stratified K-Fold: Maintains class distribution
- Leave-One-Out: K = n (expensive but low bias)
- Time Series Split: Respects temporal order
- Group K-Fold: Ensures groups don't span train/test
```

### Q7: How do you handle imbalanced datasets?

> **⚡ One-liner**: Try class weights first (free, often sufficient), then threshold tuning, then SMOTE if needed — and always evaluate with F1/PR-AUC, never plain accuracy.

**💡 Intuition**: If 99% of transactions are legit, predicting "legit" always gets 99% accuracy but catches 0 fraud. The goal is shifting the model's attention to the minority class without losing too much overall performance.

```
Strategy Ladder (try in order):
  1. Fix the metric first → swap accuracy for F1, PR-AUC, or cost-weighted metric
  2. Class weights → class_weight='balanced' (free, try this first!)
  3. Threshold tuning → move 0.5 threshold to better precision/recall balance
  4. Resampling (if above insufficient):
     - SMOTE: interpolates synthetic minority samples (k-NN based)
     - RandomUnderSampler: faster, but loses information
     - SMOTETomek: SMOTE + clean decision boundary
  5. Algorithm-level:
     - Focal loss for neural nets: FL(pt) = -α(1-pt)^γ log(pt)
     - Isolation Forest / One-class SVM for extreme imbalance
```

```python
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# Step 1: Class weights (simplest, try first)
model = LogisticRegression(class_weight='balanced')  # auto-weights minority higher
model.fit(X_train, y_train)

# Step 2: Threshold tuning (often better than resampling)
probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.3).astype(int)  # lower threshold → higher recall, lower precision
print(classification_report(y_test, preds))

# Step 3: SMOTE if still needed
smote = SMOTE(sampling_strategy=0.5, random_state=42)
# ⚠️ CRITICAL: apply SMOTE ONLY on TRAINING data, never on test!
X_res, y_res = smote.fit_resample(X_train, y_train)
```

**🚨 Common Pitfalls**:
- Applying SMOTE **before** train/test split → **data leakage**, inflated metrics
- Using ROC-AUC on 99:1 imbalance — a bad model can look great; PR-AUC is more informative
- SMOTE creates synthetic samples **in feature space**, not the actual data manifold → can add noise

**💬 Follow-ups**: "What is SMOTE?" → Interpolates between existing minority samples and their k-nearest neighbors. "Focal loss?" → Down-weights easy (confident) examples so model focuses on hard misclassifications.

### Q8: Explain PCA. What are its limitations?
```
PCA (Principal Component Analysis):
1. Center data (subtract mean)
2. Compute covariance matrix
3. Find eigenvectors (principal components)
4. Project data onto top k eigenvectors

Mathematics:
- Maximizes variance in projected space
- Components are orthogonal
- Eigenvalues indicate variance explained

Limitations:
- Linear only (use Kernel PCA for non-linear)
- Sensitive to feature scaling (standardize first)
- Components may not be interpretable
- Assumes Gaussian distribution
- May discard useful variance in lower components

When to use:
- Dimensionality reduction before ML
- Visualization (reduce to 2D/3D)
- Remove multicollinearity
- Noise reduction
```

### Q9: How does SVM work? Explain kernel trick.

> **⚡ One-liner**: SVM finds the hyperplane that maximizes the margin between classes. The kernel trick allows SVM to find non-linear boundaries by implicitly mapping data to higher dimensions without computing the mapping explicitly.

**💡 Intuition**: Imagine separating red and blue points on a table. If they're entangled, lifting the table (mapping to 3D) might make them linearly separable. The kernel trick computes the distances IN the lifted space without actually doing the lift.

```
Objective: maximize margin 2/||w||  subject to y_i(w·x_i + b) ≥ 1

Support vectors: points on or closest to the margin — ONLY these define the boundary
  → If you remove non-support-vector points, the solution doesn't change!

Soft margin (C parameter):
  Low C  → wide margin, more misclassifications OK  (high regularization)
  High C → narrow margin, enforce correct classification (low regularization)

Kernel Trick:
  K(x, x') = φ(x)·φ(x')  computed WITHOUT computing φ(x) explicitly

  Linear:     K(x,x') = x·x'                      → fast, baseline
  Polynomial: K(x,x') = (γx·x' + r)^d             → interaction features
  RBF/Gaussian: K(x,x') = exp(-γ||x-x'||²)        → most common, smooth boundary
    γ small: wide influence, smooth boundary (underfitting risk)
    γ large: narrow influence, complex boundary (overfitting risk)
```

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# SVM REQUIRES feature scaling (sensitive to feature magnitudes)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train only!
X_test_scaled  = scaler.transform(X_test)

# RBF kernel SVM with tuned hyperparameters
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.001, 0.01]}
svm = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=5)
svm.fit(X_train_scaled, y_train)
print(f"Best params: {svm.best_params_}")
```

**🚨 Common Pitfalls**:
- **Not scaling features** before SVM → RBF kernel computes distances, unscaled features dominate
- SVM scales O(n²) to O(n³) with training samples → **not suitable for large datasets** (> 100K rows); use linear SVC or neural networks instead
- High `gamma` AND high `C` together → severe overfitting, memorizes training set

**💬 Follow-ups**: "Why does SVM use support vectors and not all points?" → Only points near the boundary affect the decision; interior points provide no additional constraint. "SVM vs logistic regression?" → LR gives probabilities and scales better; SVM maximizes margin (better for small datasets with complex boundaries).

### Q10: Explain feature selection methods
```
Filter Methods (independent of model):
- Variance threshold
- Correlation with target
- Chi-squared test
- Mutual information
- Fast but ignores feature interactions

Wrapper Methods (use model):
- Recursive Feature Elimination (RFE)
- Forward/Backward selection
- Better performance but expensive

Embedded Methods (during training):
- L1 regularization (Lasso)
- Tree-based importance
- Efficient and considers interactions

Best practices:
- Start with domain knowledge
- Remove highly correlated features
- Use cross-validation when selecting
- Consider stability of selection
```

---

## Deep Learning Questions

### Q11: Explain backpropagation

> **⚡ One-liner**: Backprop is reverse-mode automatic differentiation — it computes the gradient of the loss w.r.t. every weight by applying the chain rule backwards through the computation graph in a single backward pass.

**💡 Intuition**: Each weight wants to know "how much am I responsible for the loss?" Backprop answers this by starting at the loss and distributing blame backwards — sharing intermediate computation so it's O(params) not O(params²).

```
Forward: x → [z=Wx+b] → [a=σ(z)] → [L=loss(a, y)]   store: x, z, a
Backward (chain rule, right to left):
  ∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W = δ_output · σ'(z) · x^T

Gradient Issues:
  Vanishing: sigmoid/tanh saturate → σ'(z)≈0 → gradients decay exponentially in deep nets
  Exploding: RNNs multiply gradients over T timesteps → exponential growth

Fixes:
  ✓ ReLU: gradient=1 for x>0, no saturation
  ✓ Xavier init: Var(W) = 2/(fan_in + fan_out)  → stable variance through layers
  ✓ He init:     Var(W) = 2/fan_in              → for ReLU networks
  ✓ BatchNorm / LayerNorm: re-center activations
  ✓ ResNet skip connections: gradient highway (∂(x+F(x))/∂x = 1 + ∂F/∂x)
  ✓ Gradient clipping: g = g · min(1, max_norm/||g||)
```

```python
import numpy as np

def manual_backprop(X, y, W, b, lr=0.01):
    n = X.shape[0]
    # --- FORWARD ---
    z = X @ W + b
    a = 1 / (1 + np.exp(-z))          # sigmoid
    loss = -np.mean(y*np.log(a+1e-8) + (1-y)*np.log(1-a+1e-8))
    
    # --- BACKWARD (chain rule) ---
    dL_da = -(y/(a+1e-8) - (1-y)/(1-a+1e-8)) / n
    da_dz  = a * (1 - a)               # sigmoid derivative
    dL_dz  = dL_da * da_dz
    dL_dW  = X.T @ dL_dz              # accumulate gradient from all samples
    dL_db  = np.sum(dL_dz, axis=0)
    
    W -= lr * dL_dW
    b -= lr * dL_db
    return W, b, loss
```

**🚨 Common Pitfalls**:
- Confusing backprop (computes gradients) with gradient descent (uses them)
- Forgetting backprop stores all activations → **memory scales with sequence length / depth**
- `retain_graph=True` in PyTorch needed only when calling `.backward()` multiple times on same graph

**💬 Follow-ups**: "What's automatic differentiation?" → PyTorch/JAX build a computation graph and auto-apply chain rule; backprop is just reverse-mode AD. "How does gradient checkpointing help?" → Trades compute for memory: recompute activations during backward instead of storing them.

### Q12: Compare different activation functions
```
Sigmoid: σ(x) = 1/(1+e^(-x))
- Range: (0, 1)
- Issues: Vanishing gradient, not zero-centered
- Use: Output layer for binary classification

Tanh: (e^x - e^-x)/(e^x + e^-x)
- Range: (-1, 1)
- Zero-centered but still vanishing gradient
- Use: Rarely used now

ReLU: max(0, x)
- Range: [0, ∞)
- No vanishing gradient for x > 0
- Issue: Dying ReLU (neurons output 0)
- Use: Default for hidden layers

Leaky ReLU: max(αx, x)
- Fixes dying ReLU
- α typically 0.01

GELU: x × Φ(x)
- Smooth approximation to ReLU
- Use: Transformers (BERT, GPT)

Softmax: exp(xᵢ)/Σexp(xⱼ)
- Outputs sum to 1
- Use: Multi-class classification output
```

### Q13: What is batch normalization? Why does it help?

> **⚡ One-liner**: BatchNorm normalizes each layer's inputs to zero mean/unit variance per mini-batch, then rescales with learned γ,β — stabilizing training, enabling higher learning rates, and acting as regularization.

**💡 Intuition**: Without BatchNorm, weight updates in early layers constantly shift the distribution seen by later layers (internal covariate shift) — later layers always chase a moving target. BatchNorm re-anchors each layer's input so learning is more stable.

```
Algorithm:
  μ_B = mean over batch    σ²_B = variance over batch
  x̂ = (x - μ_B) / √(σ²_B + ε)   ← normalize
  y  = γ x̂ + β                  ← learned scale+shift (allows un-normalizing)

Train vs Inference:
  Training:   use current batch μ,σ²
  Inference:  use RUNNING mean/variance (EMA accumulated during training)
  → MUST call model.eval() or BatchNorm will use batch stats at inference!

Normalization comparison:
  BatchNorm:    over (N, H, W) per channel   → needs large batch, breaks with batch_size=1
  LayerNorm:    over (C, H, W) per sample    → batch-independent, standard for Transformers
  InstanceNorm: over (H, W) per sample+ch    → removes style (used in style transfer)
  GroupNorm:    over groups of channels       → works with small batches (detection models)
  RMSNorm:      no mean centering, just scale → used in LLaMA, simpler/faster
```

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),   # normalizes over batch dimension
    nn.ReLU(),
)

# CRITICAL: switch modes!
model.train()  # BN uses batch statistics
out = model(x_batch)  # statistics computed from THIS batch

model.eval()   # BN uses RUNNING statistics (stable, deterministic)
out = model(x_batch)  # always same output for same input

# Common mistake: forgetting model.eval() at inference
# Result: different outputs each call (batch noise), worse performance
```

**🚨 Common Pitfalls**:
- Forgetting `model.eval()` at inference → BatchNorm uses live batch stats → non-deterministic, worse results
- Small batch size (< 8) with BatchNorm → noisy estimates; use GroupNorm or LayerNorm instead
- Using BatchNorm with RNNs/Transformers — use LayerNorm (batch-independent)

**💬 Follow-ups**: "Why LayerNorm in Transformers?" → Sequence lengths vary, batch sizes are small, and layer norm is batch-independent. "Does BatchNorm replace dropout?" → BN adds noise as regularization — you can reduce dropout, but they serve different purposes.

### Q14: Explain dropout. Why does it work?
```
Dropout:
- During training: randomly zero out neurons with probability p
- During inference: multiply weights by (1-p) or scale during training

Why it works:
1. Prevents co-adaptation of neurons
2. Ensemble effect (trains many sub-networks)
3. Adds noise (regularization)
4. Encourages sparse representations

Best practices:
- Typical p: 0.2-0.5
- Don't use with batch norm (both add noise)
- Apply after activation
- Higher dropout for larger layers

Variants:
- Spatial dropout: Drop entire feature maps (CNNs)
- DropConnect: Drop weights instead of activations
- DropBlock: Drop contiguous regions
```

### Q15: Explain CNN architecture components
```
Convolution layer:
- Slides filter over input
- Captures local patterns
- Parameter sharing reduces weights
- Output size: (W - K + 2P) / S + 1

Pooling layer:
- Reduces spatial dimensions
- Max pooling: Takes maximum in window
- Average pooling: Takes average
- Provides translation invariance

Stride:
- Step size of filter movement
- Stride > 1 reduces dimensions

Padding:
- 'Valid': No padding, output shrinks
- 'Same': Pad to keep same size

1x1 Convolution:
- Changes channel dimension
- Adds non-linearity
- Used in Inception, ResNet

Receptive field:
- Region of input affecting one output neuron
- Grows with depth
- Important for capturing context
```

### Q16: Explain ResNet. Why do skip connections help?
```
ResNet introduces skip/residual connections:
y = F(x) + x

Benefits:
1. Easier to learn identity mapping
   - If identity is optimal, F(x) learns 0
   
2. Gradient highway
   - Gradient flows directly through skip connection
   - Alleviates vanishing gradient
   
3. Enables very deep networks
   - Original: 152 layers
   - Without skip connections: deeper ≠ better

4. Implicit ensemble
   - Different paths through network
   - Like ensemble of shallower networks

Variants:
- Pre-activation: BN-ReLU-Conv-BN-ReLU-Conv + skip
- Bottleneck: 1x1-3x3-1x1 convolutions
- ResNeXt: Grouped convolutions
- DenseNet: Concatenate instead of add
```

### Q17: Explain LSTM. How does it solve vanishing gradient?
```
LSTM components:
- Cell state (c): Long-term memory highway
- Hidden state (h): Short-term output

Gates:
1. Forget gate: f = σ(W_f·[h_{t-1}, x_t] + b_f)
   - What to forget from cell state
   
2. Input gate: i = σ(W_i·[h_{t-1}, x_t] + b_i)
   - What new info to store
   
3. Candidate: c̃ = tanh(W_c·[h_{t-1}, x_t] + b_c)
   - New candidate values
   
4. Output gate: o = σ(W_o·[h_{t-1}, x_t] + b_o)
   - What to output

Updates:
- c_t = f * c_{t-1} + i * c̃
- h_t = o * tanh(c_t)

Why it helps vanishing gradient:
1. Additive cell state update (not multiplicative)
2. Forget gate can learn to pass gradient
3. Gradient highway through cell state
4. Gates have sigmoid (bounded gradients)
```

### Q18: Explain the attention mechanism

> **⚡ One-liner**: Attention computes a weighted sum of values, where weights come from the similarity between a query and a set of keys — letting every position directly attend to every other position in O(n²) time.

**💡 Library analogy**: Query = your question, Keys = book titles, Values = book contents. Attention lets you retrieve a weighted mixture of books proportional to how relevant each title is to your question.

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V

Why √d_k scaling?
  Dot products grow with d_k: Var[q·k] = d_k if q,k ~ N(0,1)
  Without scaling: softmax saturates (near one-hot) → vanishing gradients
  Dividing by √d_k: keeps variance at 1 regardless of d_k

Multi-Head Attention:
  Run h heads in parallel, each in d_k/h subspace
  Head 1: syntax (subject-verb)    Head 2: coreference (he → John)
  Head 3: positional proximity     Head 4: semantic similarity
  Output: Concat(head_1,...,head_h) W_O

Self vs Cross Attention:
  Self:  Q=K=V from SAME sequence    → intra-sequence dependencies
  Cross: Q from decoder, K=V from encoder → decoder attends to encoder

Complexity: O(n²·d) → quadratic in sequence length!
  → Flash Attention: tile-based computation, O(n) memory vs O(n²)
  → Sparse Attention: attend to subset of positions
```

```python
import torch, math
import torch.nn.functional as F

def attention(Q, K, V, mask=None):
    """Q, K, V: (batch, heads, seq_len, d_k)"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # scale
    if mask is not None:            # causal: block future positions
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)  # rows sum to 1
    return torch.matmul(weights, V), weights  # return weights for interpretability

# Causal mask: position i sees only positions 0..i
seq_len = 10
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
```

**🚨 Common Pitfalls**:
- Forgetting `√d_k` scaling → softmax saturation, vanishing gradients in large models
- Saying O(n log n) — standard attention is **O(n²)**; Flash Attention only reduces memory, not FLOPs
- Confusing causal mask (generation) with padding mask (variable-length batches); both needed in practice

**💬 Follow-ups**: "What is Flash Attention?" → IO-aware exact attention using tiling to avoid materializing full n×n matrix; same result, O(n) memory. "KV cache?" → During generation, reuse K,V from previous tokens instead of recomputing.

### Q19: Explain the Transformer architecture
```
Encoder:
1. Input embedding + positional encoding
2. N identical layers:
   - Multi-head self-attention
   - Add & Norm (residual connection)
   - Feed-forward network
   - Add & Norm

Decoder:
1. Output embedding + positional encoding
2. N identical layers:
   - Masked multi-head self-attention
   - Add & Norm
   - Cross-attention (Q from decoder, K,V from encoder)
   - Add & Norm
   - Feed-forward network
   - Add & Norm
3. Linear + Softmax

Key innovations:
- Self-attention replaces recurrence
- Positional encoding captures order
- Parallel processing (unlike RNN)
- Multi-head attention for multiple patterns

Why it works:
- Global receptive field from layer 1
- Direct gradient flow (residual + attention)
- Scales well with compute
```

### Q20: What are different types of normalization?
```
Batch Norm: Normalize across batch dimension
- μ, σ computed over (N, H, W) for each channel
- Good for CNNs with large batches
- Issues with small batches, RNNs

Layer Norm: Normalize across feature dimension
- μ, σ computed over (C, H, W) for each sample
- Independent of batch size
- Standard for Transformers

Instance Norm: Normalize each sample, each channel
- μ, σ computed over (H, W)
- Used in style transfer
- Removes style information

Group Norm: Normalize groups of channels
- μ, σ computed over (G, H, W) where C/G channels per group
- Works well with small batches
- Between Layer Norm and Instance Norm

RMSNorm: Simplified Layer Norm
- No mean centering, just scaling
- Used in LLaMA: RMSNorm(x) = x / RMS(x) × γ
```

---

## GenAI & LLM Questions

### Q21: Explain the difference between GPT and BERT

> **⚡ One-liner**: GPT is decoder-only with causal (left-to-right) attention, trained to predict the next token — great for generation. BERT is encoder-only with bidirectional attention, trained on masked token prediction — great for understanding tasks.

**💡 Intuition**: BERT reads the whole sentence before answering (like an open-book test). GPT generates one word at a time without peeking ahead (like writing a story).

```
                GPT (Decoder-only)      BERT (Encoder-only)     T5 (Enc-Dec)
Attention       Causal (left-only)      Bidirectional           Both
Pre-training    Next token prediction   MLM + NSP               Span corruption
Strength        Generation, few-shot    Classification, NER, QA Translation, summarization
Generate text?  Yes                     No                      Yes
Context         Only past tokens        Full sequence           Enc sees all; Dec causal
Examples        GPT-4, LLaMA, Mistral   BERT, RoBERTa           T5, FLAN-T5, BART

MLM (BERT pre-training):
  Input: "The [MASK] sat on the [MASK]"
  Target: predict "cat" and "mat"
  Why: forces model to use both left AND right context

Why GPT dominates now:
  - Scales better: decoder-only models scale to 100B+ params with stable training
  - Few-shot emergent abilities appear at scale
  - Can do understanding tasks too (just phrase as generation)
```

**🚨 Common Pitfalls**:
- Saying BERT is "better" for all NLP — GPT-4 outperforms BERT on most understanding tasks now
- Confusing "bidirectional" with "better" — BERT can't generate; GPT with enough scale is better at understanding too
- Forgetting that BERT embeddings aren't directly comparable across layers (last layer is best for most tasks)

**💬 Follow-ups**: "Why did decoder-only win?" → Simpler training objective (next token prediction scales cleanly); emergent few-shot abilities; no need for fine-tuning with RLHF. "RoBERTa vs BERT?" → RoBERTa removes NSP (was hurting, not helping), trains longer with more data.

### Q22: Explain tokenization methods (BPE, WordPiece, SentencePiece)
```
Byte Pair Encoding (BPE):
- Start with character vocabulary
- Iteratively merge most frequent pairs
- Example: "low" + "er" → "lower"
- Used by GPT, LLaMA

WordPiece:
- Similar to BPE but uses likelihood
- Merges pairs that maximize training data likelihood
- Uses "##" prefix for subwords
- Used by BERT

SentencePiece:
- Language-agnostic (no pre-tokenization)
- Treats input as raw byte stream
- Supports BPE and Unigram algorithms
- Useful for multilingual models

Unigram:
- Starts with large vocabulary
- Iteratively removes tokens to minimize loss
- More principled than BPE

Trade-offs:
- Larger vocab: Better representation, more parameters
- Smaller vocab: More tokens per text, slower
- Typical sizes: 30K-100K tokens
```

### Q23: What is LoRA? Why is it effective?

> **⚡ One-liner**: LoRA freezes the pretrained model and adds trainable low-rank matrices ΔW = B·A to each layer, reducing trainable parameters from d×k to r×(d+k) where r ≪ min(d,k) — matching full fine-tuning with ~0.1% of parameters.

**💡 Intuition**: Task-specific weight changes ΔW are empirically low-rank — they live in a small subspace. LoRA explicitly parameterizes this subspace with two thin matrices B(d×r) and A(r×k) instead of storing the full d×k update.

```
Full fine-tuning: W' = W₀ + ΔW      ← update all d×k weights
LoRA:             W' = W₀ + B·A      ← freeze W₀, train B(d×r) and A(r×k)

Parameter savings:
  d=4096, k=4096, full:     16.7M params
  d=4096, k=4096, r=16 LoRA: 2×4096×16 = 131K params  (127x fewer!)

Key design:
  A initialized: Gaussian noise   B initialized: zeros
  → At start, ΔW = BA = 0  (identical to pretrained model, stable warmup)
  Output: W₀x + (α/r)·BAx  (α scales contribution, often set to r)

Where to apply:
  Original paper: Q, V projections only
  Modern best practice: ALL linear layers (Q, K, V, O, FFN up/down) for best results

Merging for zero-latency inference:
  W_merged = W₀ + B·A  (done once after training)
  → Inference identical to original model architecture

QLoRA (Dettmers 2023):
  Base model: 4-bit NF4 quantization (NormalFloat4)
  LoRA adapters: bfloat16
  Paged optimizers: handle GPU OOM
  Result: Fine-tune 65B model on single 48GB GPU!
```

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    r=16,                                          # rank: controls capacity
    lora_alpha=32,                                 # scaling factor (alpha/r = 2)
    target_modules=["q_proj","k_proj","v_proj","o_proj"],  # which layers
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%

# After training: merge for deployment
merged = peft_model.merge_and_unload()  # W = W0 + BA, no PEFT overhead
```

**🚨 Common Pitfalls**:
- Choosing `r=4` for complex tasks — start with `r=16` and tune down if memory-constrained
- Not merging with `merge_and_unload()` before deployment → extra compute per forward pass
- Only applying LoRA to Q,V (original paper) — applying to all linear layers gives better results

**💬 Follow-ups**: "LoRA vs full fine-tuning?" → LoRA is better when compute/data are limited; full FT wins when task is very different from pretraining and you have lots of data. "DoRA?" → Decomposes W into magnitude+direction, trains both via LoRA; consistently outperforms LoRA.

### Q24: Explain RLHF (Reinforcement Learning from Human Feedback)
```
Step 1: Supervised Fine-Tuning (SFT)
- Fine-tune base model on demonstrations
- Human-written high-quality responses

Step 2: Reward Model Training
- Collect comparison data: (prompt, chosen, rejected)
- Train reward model: R(prompt, response) → score
- Loss: -log(σ(R(chosen) - R(rejected)))

Step 3: PPO Training
- Optimize policy to maximize reward
- With KL penalty to stay close to SFT model
- Objective: E[R(x,y)] - β·KL(π_θ || π_ref)

Why it works:
- Aligns model with human preferences
- Harder to write good examples than judge them
- Captures nuanced quality beyond loss functions

Alternatives:
- DPO: Direct Preference Optimization
  - No reward model needed
  - Directly optimize policy from preferences
  - Loss derived from reward model optimal policy

- Constitutional AI:
  - Model critiques and revises its own outputs
  - Uses principles instead of human feedback
```

### Q25: What is RAG? When should you use it?

> **⚡ One-liner**: RAG grounds LLM generation in retrieved documents — fetch relevant content at query time, inject it into the prompt, and generate an answer — reducing hallucinations and keeping knowledge current without retraining.

**💡 Intuition**: An LLM is an expert who only knows what was in their textbooks (training data). RAG gives them a library at query time. Better for dynamic/domain knowledge; fine-tuning is better for style/format/behavior changes.

```
RAG Pipeline:
  Query → Embed query → ANN search → top-k docs → [system + context + query] → LLM → answer

RAG vs Fine-tuning decision:
  Use RAG when:               Use Fine-tuning when:
  ✓ Knowledge changes often   ✓ Change model behavior/style
  ✓ Need citations            ✓ Static knowledge base
  ✓ Multiple knowledge bases  ✓ Latency critical (no retrieval)
  ✓ Reduce hallucinations     ✓ Deep domain reasoning

Advanced Techniques:
  Hybrid Search:  dense (embedding) + sparse (BM25) combined → better recall
  Query Rewrite:  "einstein did" → "Einstein's contributions to physics"
  Re-ranking:     cross-encoder re-scores top-100 → take top-5 (expensive but accurate)
  Iterative RAG:  partial answer → identify gaps → retrieve again
  Self-RAG:       model emits [Retrieve] token only when it needs external info
  Contextual RAG: each chunk stored with LLM-generated parent-doc context

Key metrics (RAGAS framework):
  Context Precision:  retrieved chunks are relevant (no noise)
  Context Recall:     all needed info retrieved (no gaps)
  Faithfulness:       answer grounded in context (no hallucination)
  Answer Relevancy:   answer addresses the question
```

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Minimal RAG implementation
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(chunks):
    embeddings = embedder.encode(chunks)  # (n_chunks, 384)
    return embeddings, chunks

def retrieve(query, embeddings, chunks, k=3):
    q_emb = embedder.encode([query])[0]   # (384,)
    # cosine similarity
    sims = np.dot(embeddings, q_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb))
    top_k = np.argsort(sims)[::-1][:k]
    return [chunks[i] for i in top_k]

def rag_prompt(query, context_docs):
    context = "\n\n".join(context_docs)
    return f"""Answer based ONLY on the provided context.
Say 'I don't know' if not in context.

Context:\n{context}\n\nQuestion: {query}\nAnswer:"""
```

**🚨 Common Pitfalls**:
- **Lost in the middle**: LLMs attend less to middle of context → put most relevant chunks at start or end
- **Chunking too small/large**: 256-512 tokens with 20% overlap is a good starting point
- **No re-ranking**: Top-k by cosine similarity ≠ top-k by relevance to the specific question
- **Different embedding models** for indexing vs querying → wrong similarity scores

**💬 Follow-ups**: "Multi-hop questions?" → Iterative retrieval: generate partial answer, identify gaps, retrieve again. "Embedding model choice?" → Start with `text-embedding-3-small` (OpenAI) or `BGE-M3` (open source); evaluate on your domain.

### Q26: How do you evaluate RAG systems?
```
Retrieval metrics:
- Precision@k: Relevant docs in top k
- Recall@k: Found relevant docs / total relevant
- MRR: Mean Reciprocal Rank
- NDCG: Normalized Discounted Cumulative Gain

Generation metrics:
- Faithfulness: Is answer grounded in context?
- Answer relevancy: Does answer address question?
- Answer correctness: Is answer factually correct?

RAGAS framework:
1. Context Precision: Retrieved context relevance
2. Context Recall: Coverage of ground truth
3. Faithfulness: No hallucination
4. Answer Relevancy: Answers the question

LLM-as-Judge:
- Use LLM to evaluate responses
- Criteria: helpfulness, harmlessness, honesty
- Compare to reference or pairwise comparison

Human evaluation:
- Gold standard but expensive
- Use for final validation
- Measure agreement between raters
```

### Q27: Explain prompt engineering techniques
```
Zero-shot:
"Classify sentiment: 'Great movie!' → "

Few-shot:
"Classify sentiment:
'I love it' → positive
'Terrible' → negative
'Great movie!' → "

Chain-of-Thought (CoT):
"Let's think step by step..."
- Improves reasoning tasks
- Can be zero-shot or few-shot

Self-Consistency:
- Generate multiple CoT paths
- Take majority vote
- Improves reliability

ReAct (Reasoning + Acting):
Thought: I need to find...
Action: Search[query]
Observation: Results...
Thought: Now I know...

Tree of Thoughts:
- Explore multiple reasoning paths
- Backtrack if needed
- Good for complex problems

Tips:
- Be specific and clear
- Provide examples of desired format
- Use delimiters for structure
- Specify output format (JSON, etc.)
- Include edge cases in examples
```

### Q28: What are the key challenges in deploying LLMs?
```
Latency:
- Token generation is sequential
- Solutions: KV cache, speculative decoding, smaller models

Memory:
- Large models don't fit on single GPU
- Solutions: Quantization (INT8, INT4), model parallelism

Cost:
- Inference costs scale with usage
- Solutions: Caching, batching, model distillation

Hallucination:
- Models generate false information
- Solutions: RAG, fact-checking, fine-tuning

Safety:
- Harmful outputs, prompt injection
- Solutions: Guardrails, content filtering, RLHF

Scalability:
- Handling many concurrent requests
- Solutions: Load balancing, auto-scaling, continuous batching

Monitoring:
- Detect issues in production
- Track: latency, errors, token usage, quality metrics

Updates:
- Model drift, new knowledge needed
- Solutions: RAG for dynamic info, periodic fine-tuning
```

### Q29: Explain model quantization techniques
```
Types of quantization:
- Weight-only: Quantize weights, compute in FP16/32
- Weight + Activation: Quantize both

Post-training quantization (PTQ):
- Quantize after training
- May need calibration data
- Examples: GPTQ, AWQ, bitsandbytes

Quantization-aware training (QAT):
- Simulate quantization during training
- Better accuracy but more expensive

Precision levels:
- FP16: Standard for inference
- INT8: Good accuracy, 2x speedup
- INT4: Some degradation, 4x speedup
- FP8: New GPUs support natively

GPTQ:
- Layer-wise quantization
- Uses second-order information
- Maintains good accuracy at 4-bit

AWQ (Activation-aware Weight Quantization):
- Protect important weights based on activations
- Better than uniform quantization

Trade-offs:
- Lower precision: Faster, less memory, some accuracy loss
- Quantization works better for larger models
```

### Q30: How does speculative decoding work?
```
Problem:
- LLM inference is memory-bound
- Generate one token at a time
- Most compute is reading weights from memory

Solution:
- Use small "draft" model to propose tokens
- Large "target" model verifies in parallel

Algorithm:
1. Draft model generates γ tokens quickly
2. Target model processes all γ tokens at once
3. Compare probabilities:
   - If p_target ≥ p_draft: Accept
   - Else: Accept with probability p_target/p_draft
4. Sample from adjusted distribution for rejected position

Benefits:
- Same output distribution as target model
- 2-3x speedup typical
- No accuracy degradation

Requirements:
- Draft model must be much faster
- Draft model should match target distribution well
- Works best when draft acceptance is high

Variations:
- Self-speculative: Use early layers as draft
- Medusa: Multiple prediction heads
- Lookahead: Jacobi iteration
```

---

## MLOps Questions

### Q31: Explain the ML lifecycle and MLOps
```
ML Lifecycle Stages:

1. Problem Definition
   - Business requirements
   - Success metrics
   - Constraints (latency, cost)

2. Data Collection & Preparation
   - Data sources
   - Labeling strategy
   - Data versioning

3. Feature Engineering
   - Feature extraction
   - Feature store
   - Feature validation

4. Model Development
   - Experimentation
   - Hyperparameter tuning
   - Model selection

5. Model Evaluation
   - Offline metrics
   - A/B testing
   - Shadow deployment

6. Deployment
   - Model serving
   - Scaling
   - Versioning

7. Monitoring & Maintenance
   - Performance monitoring
   - Data drift detection
   - Retraining triggers

MLOps = ML + DevOps + Data Engineering
- Automate the lifecycle
- Ensure reproducibility
- Enable collaboration
- Maintain production systems
```

### Q32: What is model drift and how do you detect it?
```
Types of Drift:

1. Data Drift (Covariate Shift)
   - Input distribution changes
   - P(X) changes, P(Y|X) same
   - Example: New user demographics

2. Concept Drift
   - Relationship between X and Y changes
   - P(Y|X) changes
   - Example: Customer preferences change

3. Label Drift
   - Output distribution changes
   - P(Y) changes
   - Example: Seasonal demand shifts

Detection Methods:

Statistical Tests:
- KS test: Compare distributions
- Chi-squared: Categorical features
- PSI (Population Stability Index)
- KL divergence

Monitoring Metrics:
- Feature statistics (mean, std, quantiles)
- Prediction distribution
- Model performance over time

PSI Formula:
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
- PSI < 0.1: No drift
- 0.1 < PSI < 0.25: Moderate drift
- PSI > 0.25: Significant drift

Response to Drift:
- Alert and investigate
- Retrain with recent data
- Update feature engineering
- Roll back if necessary
```

### Q33: Explain feature stores
```
Feature Store: Centralized repository for features

Components:
1. Online Store (low latency)
   - Redis, DynamoDB
   - Real-time serving

2. Offline Store (batch)
   - Data warehouse
   - Training data generation

3. Feature Registry
   - Metadata
   - Versioning
   - Documentation

Benefits:
- Consistency between training and serving
- Feature reuse across teams
- Point-in-time correctness
- Reduced training-serving skew

Popular Tools:
- Feast (open source)
- Tecton
- AWS Feature Store
- Databricks Feature Store

Example workflow:
1. Data scientist creates feature
2. Feature registered in store
3. Training: Pull features for date range
4. Serving: Pull features in real-time
5. Both use same feature definitions
```

### Q34: How do you version ML models and data?
```
Model Versioning:

What to track:
- Model architecture
- Hyperparameters
- Training data version
- Feature versions
- Metrics
- Code version

Tools:
- MLflow: Experiments, models, artifacts
- DVC: Data version control
- Weights & Biases: Experiment tracking
- Neptune.ai

Model Registry:
- Store trained models
- Stage management (staging → production)
- A/B testing support
- Rollback capability

Data Versioning:

Challenges:
- Large datasets
- Frequent updates
- Reproducibility

Approaches:
- DVC: Git-like for data
- Delta Lake: ACID transactions
- Immutable snapshots
- Time-travel queries

Best Practices:
- Version everything together (code, data, config)
- Automate with CI/CD
- Store lineage information
- Regular cleanup of old versions
```

### Q35: Explain model serving architectures
```
Serving Patterns:

1. Batch Prediction
   - Pre-compute predictions
   - Store in database
   - Low latency retrieval
   - Use when: Predictions don't change quickly

2. Online Prediction
   - Real-time inference
   - Synchronous API call
   - Use when: Need fresh predictions

3. Streaming Prediction
   - Process events as they arrive
   - Kafka, Flink, etc.
   - Use when: Event-driven systems

Deployment Strategies:

1. Shadow Mode
   - New model runs alongside production
   - Compare outputs, no user impact
   - Validate before rollout

2. Canary Deployment
   - Route small % to new model
   - Monitor metrics
   - Gradually increase traffic

3. Blue-Green
   - Two production environments
   - Switch traffic atomically
   - Easy rollback

4. A/B Testing
   - Split traffic randomly
   - Measure business metrics
   - Statistical significance

Serving Infrastructure:
- TensorFlow Serving
- TorchServe
- Triton Inference Server
- Seldon Core
- KServe
```

### Q36: What is CI/CD for ML?
```
CI/CD Pipeline for ML:

Continuous Integration:
1. Code changes trigger pipeline
2. Unit tests (data validation, feature tests)
3. Integration tests
4. Model training (on sample data)
5. Model validation (metrics thresholds)

Continuous Delivery:
1. Full model training
2. Model evaluation
3. Model registration
4. Staging deployment
5. Integration tests
6. Performance tests

Continuous Training:
- Automated retraining triggers
- Data freshness checks
- Drift detection
- Scheduled retraining

Pipeline Stages:
```
Code → Build → Test → Train → Evaluate → Register → Deploy → Monitor
```

Tools:
- GitHub Actions / GitLab CI
- Kubeflow Pipelines
- Airflow
- MLflow
- Argo Workflows

Testing for ML:
- Data validation tests
- Feature pipeline tests
- Model performance tests
- Integration tests
- Load tests
- Fairness tests
```

### Q37: How do you monitor ML models in production?
```
Monitoring Categories:

1. System Metrics
   - Latency (p50, p95, p99)
   - Throughput (requests/sec)
   - Error rates
   - Resource utilization

2. Data Metrics
   - Input distribution
   - Missing values
   - Feature ranges
   - Cardinality

3. Model Metrics
   - Prediction distribution
   - Confidence scores
   - Feature importance stability

4. Business Metrics
   - Conversion rate
   - Revenue impact
   - User engagement

Alerting Strategy:

Thresholds:
- Absolute: Latency > 100ms
- Relative: 2x baseline
- Trend: Increasing error rate

Alert Levels:
- Warning: Investigate soon
- Critical: Immediate action
- Page: Wake someone up

Dashboards:
- Real-time predictions
- Model performance over time
- A/B test results
- Drift indicators

Tools:
- Prometheus + Grafana
- Datadog
- Evidently AI (ML-specific)
- WhyLabs
- Arize AI
```

---

## System Design Questions

### Q38: Design a recommendation system
```
Requirements clarification:
- What are we recommending? (products, content, etc.)
- Scale: Users, items, requests per second
- Latency requirements
- Personalization level needed

Components:
1. Candidate Generation
   - Collaborative filtering (user-item interactions)
   - Content-based (item features)
   - Two-tower model (user embedding + item embedding)
   
2. Ranking
   - Use ML model with more features
   - Consider user context, item features, interaction features
   
3. Re-ranking
   - Business rules (diversity, freshness)
   - Deduplication

Data pipelines:
- Batch: Train models on historical data
- Real-time: Update user features, recent interactions

Infrastructure:
- Feature store for user/item features
- Vector database for embedding similarity
- Cache for popular items
- A/B testing framework

Evaluation:
- Offline: Precision, Recall, NDCG
- Online: CTR, engagement, revenue
```

### Q32: Design an ML system for fraud detection
```
Requirements:
- Real-time detection (low latency)
- High precision (minimize false positives)
- Handle class imbalance (fraud is rare)
- Adapt to new fraud patterns

Architecture:
1. Feature Engineering
   - User features: history, behavior patterns
   - Transaction features: amount, merchant, time
   - Aggregations: velocity, deviation from normal
   
2. Model Pipeline
   - Rules engine: Known patterns (fast, interpretable)
   - ML model: Complex patterns
   - Ensemble for final decision

3. Real-time scoring
   - Feature store for precomputed features
   - Model serving with low latency
   - Streaming aggregations (Kafka, Flink)

Handling imbalance:
- Stratified sampling
- Class weights
- Anomaly detection approach
- Focal loss

Monitoring:
- Prediction distribution drift
- Feature drift
- Feedback loop with fraud analysts

Cold start:
- Use rules initially
- Device fingerprinting
- Transfer learning from similar domains
```

### Q33: Design a search ranking system
```
Two-stage architecture:
1. Retrieval (fast, high recall)
   - Inverted index (BM25)
   - Embedding similarity (dense retrieval)
   - Hybrid approach
   
2. Ranking (slow, high precision)
   - Learning to Rank (LTR)
   - Features: query-doc match, doc quality, user context

Features:
- Query features: length, intent, entities
- Document features: PageRank, freshness, length
- Match features: BM25, embedding similarity, term coverage
- User features: history, preferences, location

Training:
- Pointwise: Predict relevance score
- Pairwise: Predict which doc is better
- Listwise: Optimize entire ranking (NDCG)

Data:
- Click data (biased toward top results)
- Human judgments (expensive but unbiased)
- Debias click data with position-aware models

Infrastructure:
- Index sharding for scale
- Caching frequent queries
- A/B testing for ranking changes
```

---

## Behavioral Questions

### Q34: Tell me about a challenging ML project

> **Framework**: STAR + TECH — Situation → Task → Action (specific!) → Result (quantified!) + Technical depth (trade-offs and WHY).

```
Answer Structure Template:

[1] Situation (20 seconds):
  "We had [business problem] affecting [scale/users] causing [quantified impact]."
  → "Our recommendation CTR was 40%, below the 60% business target."

[2] Task (10 seconds):
  "My specific role was [X], I owned [Y]."
  → "I owned feature engineering and model selection."

[3] Action (60 seconds - MOST IMPORTANT):
  "I chose X over Y because [technical trade-off reasoning]."
  "Key challenge was [problem], solved by [specific approach]."
  → "Replaced collaborative filtering with a two-tower neural model.
     Challenge: cold start for new items — solved with content embeddings.
     Chose two-tower over cross-features for serving speed —
     item embeddings can be pre-computed offline."

[4] Result (quantified!):
  "This improved [metric] from X to Y, impacting [business outcome]."
  → "CTR improved 40% → 58%, contributing $2M additional annual revenue."

[5] Technical Depth (follow-up ready):
  - What trade-offs did you make?
  - What would you do differently?
  - What was the biggest technical risk?
```

**🚨 Common Pitfalls**:
- "We improved the model" → say WHO did WHAT and HOW specifically
- No quantified result — always have at least one number
- Skipping trade-offs — interviewers want to see you considered alternatives
- Taking sole credit for team work — say "my contribution was..." not "I built everything"

### Q35: How do you stay updated with ML/AI advances?
```
Resources:
- Papers: arXiv, Papers With Code, top conferences
- Blogs: Distill, Lil'Log, Jay Alammar
- Twitter/X: Follow researchers
- Podcasts: Lex Fridman, TWIML
- Courses: Fast.ai, DeepLearning.AI
- Hands-on: Kaggle, personal projects

Strategies:
- Follow key researchers and labs
- Implement papers from scratch
- Attend meetups/conferences
- Discuss with colleagues
- Write about what you learn
```

### Q36: How do you approach debugging ML models?

> **Framework**: Data → Baseline → Overfit small set → Learning curves → Error analysis → Ablations. Always rule out the simpler cause before adding complexity.

```
Debugging Checklist (in this order):

1. VERIFY DATA FIRST (most bugs are here!)
   - Shapes, dtypes, NaN counts match expectations?
   - Label distribution: is it what you expect?
   - Train/test leakage: any future data in train?
   - Feature scaling applied consistently?
   
2. ESTABLISH A SANE BASELINE
   - Random model? Majority class? Mean prediction?
   - Rule-based system? Linear model?
   - If baseline beats your model — there's a bug
   
3. OVERFIT A SMALL DATASET (10-100 samples)
   - A well-implemented model CAN overfit a tiny dataset
   - If loss doesn't go near 0: bug in model, loss, or forward pass
   - If it overfits: model works, now add regularization
   
4. LEARNING CURVES
   - Train vs val loss over epochs and over data size
   - High bias:    both curves high   → bigger model, more features
   - High variance: gap between train/val → more data, regularization
   - Diverging loss: lr too high or wrong loss function
   
5. ERROR ANALYSIS
   - Look at worst predictions (highest loss samples)
   - Find patterns: certain categories, edge cases, distributions?
   - Confusion matrix: which classes are confused with which?
   
6. ABLATION STUDIES
   - Remove one component at a time
   - Measure impact on val metric
   - If removing a component HELPS — that component was hurting
```

```python
# Quick debugging utilities
import numpy as np

# 1. Data sanity check
def data_sanity(X_train, y_train, X_val, y_val):
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    print(f"Train labels: {np.unique(y_train, return_counts=True)}")
    print(f"Train NaNs: {np.isnan(X_train).sum()}, Val NaNs: {np.isnan(X_val).sum()}")
    print(f"Train range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    # Check for leakage: train and val should have different distributions
    print(f"Train mean: {X_train.mean():.3f}, Val mean: {X_val.mean():.3f}")

# 2. Overfit tiny dataset to verify model implementation
def verify_model(model, X_train, y_train, n_samples=32):
    X_tiny = X_train[:n_samples]
    y_tiny = y_train[:n_samples]
    # Train until convergence on tiny set
    for epoch in range(1000):
        loss = model.train_step(X_tiny, y_tiny)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {loss:.6f}")
    # Should reach near-zero loss; if not, there's a model bug
```

**🚨 Common Pitfalls**:
- Jumping to complex solutions before checking data — most ML bugs are in data pipelines
- Not establishing a baseline — hard to know if your model is actually good
- Training too long to see if the model "eventually works" — if it can't overfit 32 samples, it never will

**💬 Follow-ups**: "How do you debug NaN loss?" → Check: exploding gradients (add clipping), log of 0 in loss (add epsilon), division by 0 in normalization, incorrect masking.
```

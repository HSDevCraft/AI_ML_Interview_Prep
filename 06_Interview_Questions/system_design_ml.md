# ML System Design Interview Questions

> **Interview mindset**: ML system design is about demonstrating judgment under constraints. There's no single right answer — show you understand trade-offs, can prioritize, and think about real-world complexity (data, latency, scale, failure modes).

## Critical Interviewer Signals

| What they want to see | How to demonstrate it |
|----------------------|----------------------|
| Requirements clarity | Ask 3-4 scoping questions before diving in |
| ML judgment | Justify model choice with trade-offs, not just "I'd use XGBoost" |
| Scale awareness | Mention latency budgets, QPS, data volume proactively |
| Failure thinking | Discuss what breaks, how to detect it, how to recover |
| Iteration mindset | Start simple, then evolve — show roadmap thinking |

## Common Mistakes to Avoid

- **Jumping to models** before clarifying requirements and success metrics
- **Ignoring training data** — how will you get labels? What's the label quality?
- **Forgetting the serving gap** — features used in training must be available at inference
- **No monitoring plan** — always address: how do you know when the model breaks?
- **Single model thinking** — most production systems use multi-stage pipelines

---

## Table of Contents
1. [Framework for ML System Design](#framework-for-ml-system-design)
2. [Recommendation Systems](#recommendation-systems)
3. [Search and Ranking](#search-and-ranking)
4. [Fraud Detection](#fraud-detection)
5. [Content Moderation](#content-moderation)
6. [Ad Click Prediction](#ad-click-prediction)
7. [Entity Matching](#entity-matching)
8. [Time Series Forecasting](#time-series-forecasting)

---

## Framework for ML System Design

### Step-by-Step Approach (45-60 min interview)

```
1. CLARIFY REQUIREMENTS (5 min) — ASK BEFORE DESIGNING

   Functional clarifications:
   • "What exactly are we predicting? Who are the end users?"
   • "What's the input format? What's the expected output?"
   • "What's the current system? What are its failure modes?"

   Non-functional (scale/constraints):
   • "What's the scale? (QPS, users, items, data volume)"
   • "What's the latency budget? Real-time (<100ms) or batch?"
   • "What's the accuracy requirement? What's the cost of each error type?"
   • "How fresh does the model need to be?"

2. FRAME AS ML PROBLEM (5 min)
   • Is ML even needed? (rules/heuristics might suffice)
   • What is the prediction target? (proxy metric vs business metric)
   • Supervised / Unsupervised / Self-supervised?
   • Classification / Regression / Ranking / Generation?
   • Offline metric: what do you measure in experiments?
   • Online metric: what do you A/B test?

3. DATA (10 min) — often the most important part
   • Data sources: logs, user behavior, 3rd party
   • Label strategy: explicit ratings? clicks (noisy)? human annotation?
   • Label quality: bias, delay, sparsity
   • Features: what signals exist? what needs engineering?
   • Training/validation split: temporal split for time-series data!
   • Class imbalance, data leakage risks

4. MODEL (10 min)
   • Start with a strong baseline (logistic regression, XGBoost)
   • Explain why you'd advance to complex models
   • Discuss trade-offs explicitly:
     - Accuracy vs latency vs interpretability vs maintenance cost
     - Online learning vs periodic retraining

5. EVALUATION (5 min)
   • Offline: what metric on what dataset?
   • Online: A/B test design (randomization unit, duration, sample size)
   • Failure modes: what happens when model is wrong?
   • Guardrails: what prevents catastrophic mistakes?

6. DEPLOYMENT & SERVING (5 min)
   • Batch vs real-time inference (trade-offs!)
   • Feature store: avoid train-serving skew
   • Model versioning: canary, A/B, shadow deployment
   • Latency optimizations: caching, quantization, batching

7. MONITORING & MAINTENANCE (5 min)
   • System health: latency p99, error rate, throughput
   • Data drift: input distribution changes
   • Model drift: prediction distribution, performance degradation
   • Retraining triggers: time-based vs performance-based
   • Feedback loop: how does model affect future data?
```

### Trade-off Framework (use for every decision)

```
When choosing between options A and B, always discuss:

1. ACCURACY vs LATENCY:
   "Option A is more accurate but requires 200ms; Option B is 98% as good but runs in 20ms.
    Given our <100ms budget, I'd choose B."

2. COMPLEXITY vs MAINTAINABILITY:
   "A deep neural network would perform better, but XGBoost is:
    - Easier to debug (feature importance, SHAP values)
    - Faster to iterate on
    - More interpretable for compliance/regulation
    I'd start with XGBoost and only move to NN if it doesn't hit our metric targets."

3. FRESHNESS vs COST:
   "Real-time training gives freshest model but requires expensive streaming infra.
    Daily batch retraining covers 95% of cases at 10% of the cost.
    I'd start with daily and move to streaming only if we see significant freshness-related drift."
```

---

## Recommendation Systems

### Design: YouTube/Netflix Recommendations
```
REQUIREMENTS:
- Personalized video recommendations
- 1B+ users, 100M+ videos
- <100ms latency
- Multiple surfaces (home, search, related)

ML PROBLEM:
- Ranking problem: P(click | user, video, context)
- Objective: Maximize engagement (watch time, not just clicks)
- Multi-objective: Balance engagement, diversity, freshness

DATA:
- Implicit feedback: Views, watch time, likes, shares
- Explicit: Ratings (sparse)
- User features: Demographics, history, preferences
- Video features: Title, description, tags, embeddings
- Context: Time, device, location

ARCHITECTURE (Two-Stage):

Stage 1: Candidate Generation (retrieve 1000s from 100M+)
┌─────────────────────────────────────────────────┐
│  Multiple Sources → Merge → Dedupe             │
│  - Collaborative filtering                      │
│  - Content-based similarity                     │
│  - Popular/trending                             │
│  - Recent history                               │
└─────────────────────────────────────────────────┘

Stage 2: Ranking (score 1000s → top 100)
┌─────────────────────────────────────────────────┐
│  Deep neural network with:                      │
│  - User embedding                               │
│  - Video embedding                              │
│  - Cross features                               │
│  - Context features                             │
│  Output: P(watch), P(like), expected watch time │
└─────────────────────────────────────────────────┘

Stage 3: Re-ranking (business logic)
┌─────────────────────────────────────────────────┐
│  - Diversity (don't show same creator)          │
│  - Freshness boost                              │
│  - Policy filters                               │
│  - Ads insertion                                │
└─────────────────────────────────────────────────┘

MODELS:

Candidate Generation:
- Two-tower model: User tower + Item tower
- User tower: Embed user history → user vector
- Item tower: Embed item features → item vector
- Similarity: dot product or cosine
- ANN index for fast retrieval (FAISS, ScaNN)

Ranking:
- Wide & Deep / DCN (Deep & Cross Network)
- Multi-task learning (click, watch time, like)
- Features: user history, item features, context

EVALUATION:
Offline:
- AUC-ROC for click prediction
- NDCG for ranking quality
- Coverage, diversity metrics

Online:
- CTR (click-through rate)
- Watch time per session
- User retention

SERVING:
- Pre-compute candidate embeddings
- Cache user embeddings
- Real-time ranking with feature store
- Batch update embeddings daily

CHALLENGES:
- Cold start (new users, new items)
- Filter bubbles
- Exploitation vs exploration
- Position bias in training data
```

---

## Search and Ranking

### Design: Search Ranking System
```
REQUIREMENTS:
- E-commerce search ranking
- 1M+ products, 100M+ queries/day
- <200ms latency
- Relevant, diverse results

ML PROBLEM:
- Learning to Rank (LTR)
- Query-product relevance scoring
- Multi-objective: Relevance + Revenue

ARCHITECTURE:

Query → [Query Understanding] → [Retrieval] → [Ranking] → [Re-ranking] → Results

1. Query Understanding:
   - Spell correction
   - Query expansion (synonyms)
   - Intent classification
   - Entity extraction

2. Retrieval (1000s candidates):
   - Lexical: BM25, TF-IDF (inverted index)
   - Semantic: Dense retrieval (embeddings)
   - Hybrid: Combine both

3. Ranking (score candidates):
   - LTR model with features
   - Query-document features
   - User personalization

4. Re-ranking:
   - Diversity
   - Business rules
   - Ads

FEATURES:
Query Features:
- Query length, term frequency
- Query intent category
- Historical CTR for query

Document Features:
- Title/description relevance (BM25)
- Popularity (sales, views)
- Price, rating, reviews
- Freshness

Query-Document Features:
- Exact match in title
- BM25 score
- Semantic similarity
- Historical CTR for (query, doc) pair

User Features:
- Purchase history
- Browse history
- Demographics

LTR APPROACHES:

Pointwise:
- Predict relevance score independently
- MSE loss on relevance labels

Pairwise:
- Learn to order pairs correctly
- RankNet, LambdaRank

Listwise:
- Optimize entire ranking
- ListNet, NDCG loss

MODEL CHOICES:
- GBDT (XGBoost, LightGBM): Fast, interpretable
- Neural ranker: Better for semantic matching
- Two-tower + cross-encoder combination

EVALUATION:
Offline:
- NDCG@k, MRR, MAP
- Precision@k, Recall@k

Online:
- CTR, Add-to-cart rate
- Revenue per search
- Zero-result rate

OPTIMIZATION:
- Feature store for real-time features
- Quantized embeddings
- Caching popular queries
- Cascade ranking (cheap model filters, expensive ranks)
```

---

## Fraud Detection

### Design: Payment Fraud Detection
```
REQUIREMENTS:
- Detect fraudulent transactions in real-time
- 10M+ transactions/day
- <100ms latency
- High precision (minimize false positives)
- High recall for large amounts

ML PROBLEM:
- Binary classification: Fraud vs Legitimate
- Extreme class imbalance (0.1% fraud)
- Real-time inference with streaming features

DATA:
Transaction Features:
- Amount, merchant, category
- Payment method, device
- Time, location

User Features:
- Account age, history
- Average transaction amount
- Typical merchants, locations
- Velocity (transactions per hour)

Derived Features:
- Deviation from user's average
- Time since last transaction
- New merchant flag
- Distance from last location

Labels:
- Chargebacks (delayed feedback)
- Manual review decisions
- User reports

ARCHITECTURE:

Transaction → [Feature Computation] → [Rules] → [ML Model] → [Decision]
                                         ↓
                                    [Queue for Review]

1. Rules Engine (fast, interpretable):
   - Block known fraud patterns
   - Velocity limits
   - Blacklisted merchants

2. ML Model:
   - Score transaction risk
   - Combine with rules

3. Decision Logic:
   - Low risk: Approve
   - High risk: Block
   - Medium: Queue for review

MODELS:

Baseline: Logistic Regression + Rules
- Interpretable
- Fast inference
- Good for simple patterns

Advanced: Gradient Boosting (XGBoost/LightGBM)
- Better accuracy
- Handles feature interactions
- Feature importance for explainability

Deep Learning: For sequence patterns
- User transaction sequences
- LSTM/Transformer for behavior modeling

Graph Neural Networks:
- Model relationships (user-merchant-device)
- Detect fraud rings

HANDLING IMBALANCE:
- Class weights
- Oversampling (SMOTE)
- Undersampling
- Focal loss
- Anomaly detection approach

EVALUATION:
Offline:
- Precision-Recall curve (not ROC for imbalanced)
- F1 at various thresholds
- Cost-weighted metrics

Online:
- Fraud loss rate
- False positive rate
- Review queue size

CHALLENGES:
- Adversarial environment (fraudsters adapt)
- Delayed labels (chargebacks take weeks)
- Cold start (new users)
- Explainability requirements

MONITORING:
- Score distribution shift
- Feature drift
- Fraud rate by segment
- Model refresh triggers
```

---

## Content Moderation

### Design: Toxic Content Detection
```
REQUIREMENTS:
- Detect harmful content (hate speech, violence, spam)
- User-generated text, images, videos
- <500ms for text, longer for media
- High recall (catch most bad content)

ML PROBLEM:
- Multi-label classification
- Categories: Hate, Violence, Sexual, Spam, etc.
- Severity scoring

ARCHITECTURE:

Content → [Pre-filter] → [ML Models] → [Aggregation] → [Action]
                              ↓
                        [Human Review Queue]

Actions:
- Approve
- Remove
- Reduce distribution
- Add warning label
- Queue for human review

TEXT MODERATION:

Models:
- Fine-tuned BERT/RoBERTa
- Multi-task for different categories
- Severity regression head

Challenges:
- Context matters (news vs. threat)
- Evolving slang and codes
- Multilingual content
- Sarcasm, satire

Features:
- Text embeddings
- User history
- Context (post, comment, DM)
- Reported content patterns

IMAGE MODERATION:

Models:
- CNN classifiers (ResNet, EfficientNet)
- Multi-label classification
- Object detection for specific content

Challenges:
- Memes with text overlay
- Artistic content vs. explicit
- Manipulated images

VIDEO MODERATION:

Approach:
1. Sample frames
2. Classify each frame
3. Aggregate scores
4. Audio analysis separately

Models:
- Frame-level CNN
- Video transformers (expensive)
- Audio classifier

HUMAN-IN-THE-LOOP:

Review Queue Priority:
- High severity, borderline confidence
- Viral content
- Appeals

Feedback Loop:
- Human decisions → retrain model
- Update guidelines
- Calibrate thresholds

EVALUATION:
Offline:
- Precision/Recall per category
- False positive rate (user experience)
- False negative rate (safety)

Online:
- Appeal rate
- Reviewer agreement
- Time to action

CHALLENGES:
- Subjectivity in guidelines
- Cultural context
- Evolving threats
- Adversarial attacks (bypass attempts)
- Scale (billions of posts)
```

---

## Ad Click Prediction

### Design: Ad CTR Prediction
```
REQUIREMENTS:
- Predict P(click | user, ad, context)
- 10B+ predictions/day
- <10ms latency
- Maximize revenue while maintaining relevance

ML PROBLEM:
- Binary classification (click/no-click)
- Extremely sparse, high-dimensional features
- Real-time serving

DATA:
User Features:
- Demographics
- Interest categories
- Browse/purchase history
- Device, location

Ad Features:
- Creative (image, text)
- Advertiser, category
- Landing page quality
- Historical CTR

Context Features:
- Placement (position, page type)
- Time of day, day of week
- Previous ads shown

Labels:
- Clicks (positive)
- Impressions without click (negative)
- Post-click conversions (for pCVR)

ARCHITECTURE:

Request → [Feature Store] → [Model] → [Bid Calculation] → [Auction] → [Ad Shown]
                                              ↑
                                        Revenue = CTR × Bid × CVR

MODELS:

Logistic Regression (Baseline):
- Fast, interpretable
- Feature crosses manually
- FTRL for online learning

Factorization Machines:
- Automatic feature interactions
- Handles sparse features

Deep Learning:
- Wide & Deep
- DeepFM (FM + DNN)
- DCN (Deep & Cross Network)
- DLRM (Facebook's approach)

Architecture (DeepFM):
┌─────────────────────────────────────────┐
│  Sparse Features → Embeddings           │
│         ↓                               │
│  ┌─────────────┐  ┌─────────────┐       │
│  │ FM Component │  │ DNN Component│      │
│  │ (interactions)│ │ (deep)       │      │
│  └─────────────┘  └─────────────┘       │
│         ↓                ↓              │
│         └───── Concat ────┘             │
│                  ↓                      │
│              Sigmoid → CTR              │
└─────────────────────────────────────────┘

TRAINING:
- Cross-entropy loss
- Importance sampling (subsample negatives)
- Distributed training
- Frequent updates (hourly/daily)

SERVING OPTIMIZATION:
- Embed lookup caching
- Quantized embeddings
- Feature hashing for long-tail
- Approximate calculations

EVALUATION:
Offline:
- AUC-ROC
- Log loss (calibration matters!)
- Normalized cross-entropy (NCE)

Online:
- CTR, CVR
- Revenue per 1000 impressions (RPM)
- Advertiser ROI

CALIBRATION:
- Important for auction pricing
- Platt scaling
- Isotonic regression
- Temperature scaling
```

---

## Entity Matching

### Design: Product Deduplication
```
REQUIREMENTS:
- Match same products across different sellers
- 100M+ product listings
- Batch processing (daily)
- High precision (wrong matches cause problems)

ML PROBLEM:
- Pairwise classification: Same product or not
- Blocking to reduce comparisons
- Clustering for final groups

ARCHITECTURE:

Products → [Blocking] → [Candidate Pairs] → [Matching] → [Clustering] → [Canonical Products]

1. Blocking (reduce O(n²)):
   - Rule-based: Same brand + category
   - LSH (Locality Sensitive Hashing)
   - Embedding similarity threshold

2. Pairwise Matching:
   - Compare candidate pairs
   - Output: Match probability

3. Clustering:
   - Group matched pairs
   - Handle transitivity

FEATURES FOR MATCHING:

Text Similarity:
- Title: Jaccard, cosine, edit distance
- Description similarity
- TF-IDF vectors

Attribute Matching:
- Brand exact match
- Category overlap
- Price similarity
- Dimension similarity

Embedding Similarity:
- BERT embeddings of title
- Image embeddings (if available)

MODELS:

Traditional:
- Feature engineering + XGBoost
- Interpretable, adjustable

Deep Learning:
- Siamese networks
- BERT for text matching
- Cross-encoder for high accuracy

Training Data:
- Manual labeled pairs
- Semi-supervised (high-confidence matches)
- Active learning for hard cases

BLOCKING STRATEGIES:

Rule-based:
- Same (normalized brand, first word of title)
- Pros: Fast, interpretable
- Cons: May miss variations

Embedding-based:
- Cluster similar embeddings
- ANN search (FAISS)
- Pros: Catches variations
- Cons: May include false positives

Hybrid:
- Rules for high precision
- Embeddings for coverage

EVALUATION:
- Precision: Matched pairs that are correct
- Recall: True matches found
- Cluster purity
- F1 at threshold

CHALLENGES:
- Variations in titles ("iPhone 12" vs "Apple iPhone 12 64GB")
- Missing attributes
- Seller-specific additions
- Multi-pack vs single
- Counterfeit products
```

---

## Time Series Forecasting

### Design: Demand Forecasting System
```
REQUIREMENTS:
- Forecast product demand for inventory
- 1M+ products, 1000+ locations
- Multiple horizons (1 day, 1 week, 1 month)
- Update daily

ML PROBLEM:
- Multi-horizon time series forecasting
- Hierarchical (product → category → total)
- Probabilistic (uncertainty quantification)

DATA:
Historical:
- Sales by product, location, date
- Stock levels (for censored demand)
- Returns

Features:
- Calendar: Day of week, holidays, events
- Price: Current, promotions, competitor
- External: Weather, economic indicators
- Product: Category, lifecycle stage

Hierarchy:
Total → Region → Store → Category → Product

ARCHITECTURE:

Historical Data → [Feature Engineering] → [Models] → [Reconciliation] → [Forecasts]

1. Feature Engineering:
   - Lag features (sales t-1, t-7, t-30)
   - Rolling statistics (mean, std)
   - Fourier features for seasonality
   - Holiday encodings

2. Models (ensemble):
   - Statistical baselines
   - ML models
   - Combine predictions

3. Reconciliation:
   - Ensure hierarchy consistency
   - Bottom-up, top-down, or optimal

MODELS:

Statistical:
- ARIMA, SARIMA
- Exponential smoothing (ETS)
- Prophet (Facebook)
- Good for individual series

ML:
- LightGBM with lag features
- Good for many similar series
- Handles external features well

Deep Learning:
- DeepAR (Amazon)
- N-BEATS
- Temporal Fusion Transformer
- Good for large datasets, probabilistic

Global vs Local:
- Local: One model per series (ARIMA)
- Global: One model, all series (DeepAR)
- Global learns cross-series patterns

PROBABILISTIC FORECASTS:
- Quantile regression
- MC Dropout
- Conformal prediction

Output: 10th, 50th, 90th percentiles
- Safety stock based on 90th percentile
- Planning based on median

EVALUATION:
Point Forecasts:
- MAPE, SMAPE
- RMSE
- Bias

Probabilistic:
- Pinball loss (quantile)
- Coverage
- Calibration

Business Metrics:
- Stockout rate
- Overstock cost
- Service level

CHALLENGES:
- Intermittent demand (many zeros)
- New products (cold start)
- Promotions (demand spikes)
- Stock-outs (censored data)
- Hierarchy consistency

DEPLOYMENT:
- Batch predictions daily/weekly
- Store in database
- Dashboard for planners
- Alerts for unusual patterns
```

---

## Interview Tips

### Common Follow-up Questions
```
SCALE:
"What if we have 10x more users?"
- Discuss sharding, caching
- Model compression
- Feature selection

LATENCY:
"What if we need 10ms latency?"
- Pre-computation
- Model distillation
- Approximate algorithms
- Caching

COLD START:
"What about new users/items?"
- Content-based fallback
- Popular items
- Explore/exploit
- Progressive personalization

FAIRNESS:
"How do you ensure fairness?"
- Define fairness metrics
- Audit for bias
- Debiasing techniques

EXPLAINABILITY:
"How would you explain predictions?"
- SHAP values
- Feature importance
- Simpler interpretable model

ONLINE LEARNING:
"What if data changes frequently?"
- Incremental updates
- Online learning algorithms
- Frequent retraining

FAILURE MODES:
"What happens if the model fails?"
- Fallback rules
- Graceful degradation
- Monitoring and alerts
```

### Questions to Ask Interviewer
```
- What's the most important metric to optimize?
- What latency constraints do we have?
- How much labeled data is available?
- Is there an existing system to improve?
- What's the deployment environment?
- Are there regulatory/fairness requirements?
```

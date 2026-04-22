# ML/DL/GenAI Interview Questions Bank

## Machine Learning Questions

### Q1: Explain Bias-Variance Tradeoff
```
Bias: Error from erroneous assumptions (underfitting)
- High bias = model too simple
- Misses relevant relations

Variance: Error from sensitivity to small fluctuations (overfitting)
- High variance = model too complex
- Captures noise as if it were signal

Total Error = Bias² + Variance + Irreducible Error

Solutions:
- High Bias: More complex model, more features, less regularization
- High Variance: More data, simpler model, regularization, dropout, ensemble
```

### Q2: What is regularization? Compare L1 vs L2
```
Regularization adds penalty to loss function to prevent overfitting.

L1 (Lasso): Loss + λ Σ|w|
- Produces sparse weights (feature selection)
- Some weights become exactly 0
- Good when few features are important

L2 (Ridge): Loss + λ Σw²
- Produces small but non-zero weights
- Distributes weight among correlated features
- Better for preventing overfitting in general

Elastic Net: Combines both
Loss + λ₁ Σ|w| + λ₂ Σw²
```

### Q3: Explain precision, recall, F1-score. When to use each?
```
Precision = TP / (TP + FP)
- "Of predicted positives, how many are correct?"
- Important when false positives are costly (spam detection)

Recall = TP / (TP + FN)
- "Of actual positives, how many did we find?"
- Important when false negatives are costly (disease detection)

F1 = 2 × (Precision × Recall) / (Precision + Recall)
- Harmonic mean, balances both
- Good for imbalanced datasets

Use Cases:
- Spam: High precision (don't want to lose important emails)
- Cancer: High recall (don't want to miss cases)
- Balanced importance: F1 score
```

### Q4: How does Random Forest work? Why is it effective?
```
Random Forest:
1. Create multiple decision trees (ensemble)
2. Each tree trained on bootstrap sample (bagging)
3. At each split, consider random subset of features
4. Final prediction: majority vote (classification) or average (regression)

Why effective:
- Reduces variance through averaging
- Decorrelates trees via feature randomization
- Robust to outliers and noise
- Handles high-dimensional data well
- Provides feature importance

Key hyperparameters:
- n_estimators: Number of trees
- max_depth: Tree depth limit
- max_features: Features per split (sqrt for classification, n/3 for regression)
- min_samples_split: Minimum samples to split node
```

### Q5: Explain gradient boosting. How does XGBoost improve on it?
```
Gradient Boosting:
1. Start with simple model (e.g., mean)
2. Compute residuals (errors)
3. Fit new model to predict residuals
4. Add new model to ensemble (with learning rate)
5. Repeat

XGBoost improvements:
- Regularization: L1 and L2 on leaf weights
- Second-order gradient: Uses Hessian for better approximation
- Column subsampling: Like Random Forest
- Efficient handling of sparse data
- Parallel processing at tree level
- Built-in cross-validation
- Early stopping

Loss: Σ l(yᵢ, ŷᵢ) + Σ Ω(fₖ)
where Ω(f) = γT + ½λ||w||²
```

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
```
Data-level:
- Oversampling minority (SMOTE, ADASYN)
- Undersampling majority
- Combination (SMOTETomek)

Algorithm-level:
- Class weights (class_weight='balanced')
- Cost-sensitive learning
- Anomaly detection approach

Evaluation:
- Don't use accuracy alone
- Use precision, recall, F1, AUC-ROC
- PR-AUC for highly imbalanced data

Model selection:
- Ensemble methods often work better
- Adjust decision threshold
- Focal loss for neural networks
```

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
```
SVM (Support Vector Machine):
- Find hyperplane that maximizes margin between classes
- Support vectors: points closest to decision boundary

For non-linearly separable data:
- Map to higher dimensional space
- Find linear separator in that space

Kernel Trick:
- Compute dot products in high-dim space without explicit mapping
- K(x, y) = φ(x) · φ(y)

Common kernels:
- Linear: K(x,y) = x·y
- Polynomial: K(x,y) = (γx·y + r)^d
- RBF: K(x,y) = exp(-γ||x-y||²)

Key parameters:
- C: Regularization (lower = more regularization)
- gamma: RBF kernel width (higher = more complex boundary)
```

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
```
Backpropagation: Algorithm to compute gradients for neural network training

Forward pass:
- Compute activations layer by layer
- Store intermediate values

Backward pass:
- Compute loss gradient w.r.t. output
- Propagate gradient backward using chain rule
- ∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w

Key insights:
- Chain rule enables efficient gradient computation
- Compute gradients in O(n) vs O(n²) for numerical differentiation
- Need to store activations for backward pass

Gradient flow issues:
- Vanishing: gradients → 0 (deep networks, sigmoid)
- Exploding: gradients → ∞ (RNNs)

Solutions:
- Proper initialization (Xavier, He)
- Batch/Layer normalization
- Skip connections (ResNet)
- Gradient clipping
```

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
```
Batch Normalization:
1. Normalize activations: x̂ = (x - μ_batch) / σ_batch
2. Scale and shift: y = γx̂ + β (learnable)

Benefits:
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularization (due to batch statistics)
- Reduces sensitivity to initialization

Where to apply:
- After linear/conv layer, before activation
- Or after activation (both work)

Issues:
- Depends on batch size
- Different behavior train vs inference
- Problematic for RNNs (use Layer Norm)

Alternatives:
- Layer Norm: Normalize across features (for transformers)
- Instance Norm: Per-sample, per-channel (for style transfer)
- Group Norm: Groups of channels (for small batches)
```

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
```
Attention: Weighted sum of values based on query-key similarity

Scaled Dot-Product Attention:
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Components:
- Query (Q): What we're looking for
- Key (K): What we match against
- Value (V): What we retrieve
- √d_k: Scaling factor for stable gradients

Multi-Head Attention:
- Run attention multiple times in parallel
- Each head learns different relationships
- Concatenate and project results

Self-Attention:
- Q, K, V all come from same sequence
- Each position attends to all positions
- Captures long-range dependencies

Benefits over RNN:
- Parallel computation
- Direct connections (no vanishing gradient)
- O(1) path length between positions

Complexity: O(n² × d) for sequence length n
```

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
```
GPT (Generative Pre-trained Transformer):
- Decoder-only architecture
- Causal/autoregressive: predicts next token
- Pre-training: Language modeling (next token prediction)
- Unidirectional attention (left-to-right)
- Best for: Generation, few-shot learning

BERT (Bidirectional Encoder Representations):
- Encoder-only architecture
- Pre-training: MLM + NSP
  - Masked Language Modeling (15% tokens)
  - Next Sentence Prediction
- Bidirectional attention
- Best for: Classification, NER, QA

Key trade-offs:
- BERT: Better understanding, can't generate
- GPT: Better generation, unidirectional context
- T5: Encoder-decoder, text-to-text for all tasks
```

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
```
LoRA (Low-Rank Adaptation):
- Instead of fine-tuning all weights W
- Train low-rank decomposition: ΔW = BA
- B ∈ R^(d×r), A ∈ R^(r×k) where r << min(d,k)
- Output: W₀x + BAx

Benefits:
1. Memory efficient
   - Only store r×(d+k) params vs d×k
   - Typical r = 8-64
   
2. No inference latency
   - Merge: W' = W₀ + BA
   - Same architecture as original
   
3. Task switching
   - Keep base model, swap LoRA weights
   - Multiple tasks with minimal storage
   
4. Prevents catastrophic forgetting
   - Original weights frozen
   - Small adaptation maintains knowledge

Applied to:
- Query and Value projections typically
- Sometimes all linear layers

QLoRA: LoRA + 4-bit quantization
- Base model quantized to 4-bit
- LoRA weights in higher precision
- Enables 7B model fine-tuning on consumer GPU
```

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
```
RAG (Retrieval-Augmented Generation):
1. Retrieve: Find relevant documents for query
2. Augment: Add documents to prompt context
3. Generate: LLM generates answer using context

Components:
- Embedding model: Convert text to vectors
- Vector database: Store and search embeddings
- LLM: Generate final response

When to use RAG:
✓ Need up-to-date information
✓ Domain-specific knowledge
✓ Need citations/sources
✓ Reduce hallucinations
✓ Keep model weights fixed

When NOT to use RAG:
✗ Real-time latency critical
✗ Simple tasks model already knows
✗ No relevant documents exist
✗ Creative generation tasks

Advanced techniques:
- Hybrid search (dense + sparse)
- Query rewriting/expansion
- Re-ranking retrieved documents
- Iterative retrieval
- Self-RAG (model decides when to retrieve)
```

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

## System Design Questions

### Q31: Design a recommendation system
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
```
STAR Format:
- Situation: What was the context/problem?
- Task: What was your specific responsibility?
- Action: What did you do? (Be specific about your contribution)
- Result: What was the outcome? (Quantify if possible)

Key points to cover:
- Technical challenges and how you overcame them
- Collaboration with team members
- Trade-offs you considered
- Learnings from the project
- Impact on business/users
```

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
```
Systematic approach:
1. Verify data pipeline
   - Check for data leaks
   - Validate preprocessing
   - Look at data distribution
   
2. Start simple
   - Baseline model first
   - Add complexity gradually
   
3. Overfit small dataset
   - If can't overfit, bug in model/training
   - Then add regularization
   
4. Learning curves
   - High bias: Need more capacity
   - High variance: Need more data/regularization
   
5. Error analysis
   - Look at failure cases
   - Find patterns in errors
   - Feature importance
   
6. Ablation studies
   - Remove components one by one
   - Understand contribution of each part
```

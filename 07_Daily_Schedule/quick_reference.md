# Quick Reference Cards

## DSA Pattern Templates

### Two Pointers
```python
def two_pointers(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        # Process based on condition
        if condition:
            left += 1
        else:
            right -= 1
```

### Sliding Window (Variable Size)
```python
def sliding_window(s):
    left = 0
    window = {}  # or set, Counter
    result = 0
    
    for right in range(len(s)):
        # Expand window
        window[s[right]] = window.get(s[right], 0) + 1
        
        # Contract window when invalid
        while invalid_condition:
            window[s[left]] -= 1
            left += 1
        
        # Update result
        result = max(result, right - left + 1)
    
    return result
```

### Binary Search
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # or left for insertion point
```

### BFS Template
```python
from collections import deque

def bfs(graph, start):
    visited = {start}
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### DFS Template
```python
def dfs(graph, node, visited):
    if node in visited:
        return
    visited.add(node)
    
    for neighbor in graph[node]:
        dfs(graph, neighbor, visited)
```

### Backtracking Template
```python
def backtrack(candidates, path, result, start):
    if is_solution(path):
        result.append(path[:])
        return
    
    for i in range(start, len(candidates)):
        path.append(candidates[i])
        backtrack(candidates, path, result, i + 1)  # or i for reuse
        path.pop()
```

### DP Template
```python
def dp_solution(input):
    n = len(input)
    dp = [0] * (n + 1)  # or 2D: [[0]*(m+1) for _ in range(n+1)]
    
    # Base cases
    dp[0] = base_value
    
    # Fill DP table
    for i in range(1, n + 1):
        dp[i] = recurrence_relation(dp, i)
    
    return dp[n]
```

### Union-Find
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```

---

## ML Formulas Cheat Sheet

### Linear Regression
```
y = Xw + b
Loss: MSE = (1/n) Σ(y - ŷ)²
Gradient: ∂L/∂w = (2/n) X^T(Xw - y)
Normal Equation: w = (X^T X)^(-1) X^T y
```

### Logistic Regression
```
σ(z) = 1 / (1 + e^(-z))
p = σ(Xw + b)
Loss: BCE = -Σ[y·log(p) + (1-y)·log(1-p)]
```

### Regularization
```
L1 (Lasso): Loss + λ Σ|w|
L2 (Ridge): Loss + λ Σw²
ElasticNet: Loss + λ₁ Σ|w| + λ₂ Σw²
```

### Metrics
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Information Theory
```
Entropy: H(X) = -Σ p(x) log p(x)
Cross-Entropy: H(p,q) = -Σ p(x) log q(x)
KL Divergence: D_KL(p||q) = Σ p(x) log(p(x)/q(x))
```

### Gradient Descent
```
w = w - η × ∂L/∂w

SGD: Update per sample
Mini-batch: Update per batch
Adam: m = β₁m + (1-β₁)g
      v = β₂v + (1-β₂)g²
      w = w - η × m̂ / (√v̂ + ε)
```

---

## Deep Learning Quick Reference

### Activation Functions
| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| Sigmoid | 1/(1+e^(-x)) | (0,1) | Binary output |
| Tanh | (e^x-e^(-x))/(e^x+e^(-x)) | (-1,1) | Hidden layers |
| ReLU | max(0,x) | [0,∞) | Default hidden |
| LeakyReLU | max(αx,x) | (-∞,∞) | Avoid dying ReLU |
| GELU | x·Φ(x) | (-∞,∞) | Transformers |
| Softmax | e^xi/Σe^xj | (0,1) | Multi-class output |

### CNN Formulas
```
Output size = (W - K + 2P) / S + 1

W = input size
K = kernel size
P = padding
S = stride
```

### Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Multi-head: Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Transformer Components
```
LayerNorm(x + Sublayer(x))

FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

Positional Encoding:
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

---

## GenAI Quick Reference

### Tokenization
```
BPE: Merge most frequent pairs iteratively
WordPiece: Merge pairs maximizing likelihood
SentencePiece: Language-agnostic, treats as bytes
```

### Fine-tuning Methods
| Method | Params Trained | Memory | Use Case |
|--------|----------------|--------|----------|
| Full | All | High | Sufficient data/compute |
| LoRA | ~0.1% | Low | Most cases |
| QLoRA | ~0.1% | Very Low | Limited GPU |
| Prefix | ~1% | Medium | Few-shot tasks |

### LoRA
```
W' = W + BA
B ∈ R^(d×r), A ∈ R^(r×k)
r << min(d, k)
```

### RAG Pipeline
```
Query → Embed → Vector Search → Retrieve Docs → 
Augment Prompt → LLM → Response
```

### Evaluation Metrics
```
Perplexity = exp(avg cross-entropy loss)
BLEU: n-gram precision
ROUGE: n-gram recall
BERTScore: Semantic similarity
```

---

## Complexity Cheat Sheet

### Time Complexity
| Operation | Array | Linked List | BST (balanced) | Hash Table |
|-----------|-------|-------------|----------------|------------|
| Access | O(1) | O(n) | O(log n) | O(1) |
| Search | O(n) | O(n) | O(log n) | O(1) |
| Insert | O(n) | O(1) | O(log n) | O(1) |
| Delete | O(n) | O(1) | O(log n) | O(1) |

### Sorting Algorithms
| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No |
| Tim Sort | O(n) | O(n log n) | O(n log n) | O(n) | Yes |

### Graph Algorithms
| Algorithm | Time | Space |
|-----------|------|-------|
| BFS/DFS | O(V + E) | O(V) |
| Dijkstra | O((V + E) log V) | O(V) |
| Bellman-Ford | O(VE) | O(V) |
| Floyd-Warshall | O(V³) | O(V²) |
| Topological Sort | O(V + E) | O(V) |

---

## Interview Day Checklist

### Before Interview
- [ ] Review key algorithms and patterns
- [ ] Review project descriptions (STAR)
- [ ] Prepare questions for interviewer
- [ ] Test audio/video setup
- [ ] Have water and snacks ready
- [ ] Get good sleep night before

### During Coding Interview
1. **Clarify** (2 min): Ask questions, confirm understanding
2. **Example** (2 min): Work through example
3. **Approach** (5 min): Explain your approach before coding
4. **Code** (15-20 min): Write clean code
5. **Test** (5 min): Walk through test cases
6. **Optimize** (5 min): Discuss improvements

### STAR Format for Behavioral
- **S**ituation: Context and background
- **T**ask: Your responsibility
- **A**ction: What you did specifically
- **R**esult: Outcome with metrics

### Questions to Ask Interviewer
- What does a typical day look like?
- What are the biggest challenges the team faces?
- How do you measure success in this role?
- What's the tech stack/ML infrastructure?
- What opportunities for growth exist?

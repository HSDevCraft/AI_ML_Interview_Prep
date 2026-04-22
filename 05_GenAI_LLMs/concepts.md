# Generative AI & Large Language Models - Complete Guide

## 1. LLM Fundamentals

### Architecture Evolution

```
Timeline of Key Models:
- 2017: Transformer (Attention Is All You Need)
- 2018: GPT-1 (117M params), BERT (340M params)
- 2019: GPT-2 (1.5B params), T5, XLNet
- 2020: GPT-3 (175B params)
- 2022: ChatGPT, InstructGPT
- 2023: GPT-4, LLaMA, Claude, Mistral
- 2024: LLaMA 3, Claude 3, Gemini, GPT-4o
```

### GPT Architecture (Decoder-Only)

```python
class GPTBlock(nn.Module):
    """GPT-style decoder block"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_ff),
            nn.GELU(),
            nn.Linear(4 * d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Pre-norm architecture (used in GPT-2+)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalSelfAttention(nn.Module):
    """Masked self-attention for autoregressive generation"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.size()
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.d_k)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, T, d_k]
        
        # Attention with causal mask
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Causal mask: prevent attending to future tokens
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v  # [B, heads, T, d_k]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.proj(y)


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, 
                 num_layers=12, max_len=1024, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_model, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.token_emb.weight = self.head.weight
    
    def forward(self, idx):
        B, T = idx.size()
        
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
```

### BERT Architecture (Encoder-Only)

```python
class BERTEmbedding(nn.Module):
    """BERT combines token, segment, and position embeddings"""
    def __init__(self, vocab_size, d_model, max_len, dropout=0.1):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.segment = nn.Embedding(2, d_model)  # Sentence A/B
        self.position = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tokens, segments):
        seq_len = tokens.size(1)
        positions = torch.arange(seq_len, device=tokens.device)
        
        x = self.token(tokens) + self.segment(segments) + self.position(positions)
        return self.dropout(self.norm(x))


# BERT Pre-training Objectives:
# 1. Masked Language Modeling (MLM)
#    - Randomly mask 15% of tokens
#    - 80% [MASK], 10% random, 10% unchanged
#    - Predict original tokens

# 2. Next Sentence Prediction (NSP)
#    - Binary classification: is sentence B the next sentence after A?
#    - 50% actual next, 50% random

# Fine-tuning for different tasks:
# - Classification: [CLS] token output
# - NER: Per-token classification
# - QA: Start/end span prediction
```

### Model Comparison

| Model | Type | Params | Context | Use Case |
|-------|------|--------|---------|----------|
| BERT | Encoder | 110M-340M | 512 | Classification, NER |
| GPT-3 | Decoder | 175B | 4K | Generation, Few-shot |
| T5 | Enc-Dec | 60M-11B | 512 | Text-to-text |
| LLaMA 2 | Decoder | 7B-70B | 4K | Open-source LLM |
| GPT-4 | Decoder | ~1.7T? | 128K | Multimodal, Reasoning |

---

## 2. Tokenization

### Tokenization Methods

```python
# 1. Word-level Tokenization
text = "I love machine learning"
tokens = text.split()  # ['I', 'love', 'machine', 'learning']
# Problem: Large vocabulary, OOV words

# 2. Character-level Tokenization
tokens = list(text)  # ['I', ' ', 'l', 'o', 'v', 'e', ...]
# Problem: Very long sequences, loses word meaning

# 3. Subword Tokenization (BPE, WordPiece, SentencePiece)
# Balance between word and character level
```

### Byte Pair Encoding (BPE)

```python
class BPETokenizer:
    """Simplified BPE implementation"""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
    
    def get_stats(self, ids):
        """Count frequency of adjacent pairs"""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids, pair, new_id):
        """Merge all occurrences of pair into new_id"""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def train(self, text):
        """Train BPE on text"""
        # Start with character-level tokens
        ids = list(text.encode('utf-8'))
        
        # Initialize vocab with bytes
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        num_merges = self.vocab_size - 256
        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            
            # Find most frequent pair
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            
            # Merge
            ids = self.merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        
        return ids
    
    def encode(self, text):
        """Encode text to token ids"""
        ids = list(text.encode('utf-8'))
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            ids = self.merge(ids, pair, self.merges[pair])
        return ids
    
    def decode(self, ids):
        """Decode token ids to text"""
        tokens = b''.join(self.vocab[id] for id in ids)
        return tokens.decode('utf-8', errors='replace')


# Using Hugging Face tokenizers
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)

# Special tokens
# [CLS], [SEP], [PAD], [MASK], [UNK]
# <s>, </s>, <pad>, <unk> (for GPT/LLaMA)
```

---

## 3. Training LLMs

### Pre-training Objectives

```python
# 1. Causal Language Modeling (CLM) - GPT style
# Predict next token given previous tokens
# P(x_t | x_1, ..., x_{t-1})

def causal_lm_loss(logits, labels):
    """
    logits: [batch, seq_len, vocab_size]
    labels: [batch, seq_len] (shifted by 1)
    """
    # Shift so we predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    return loss


# 2. Masked Language Modeling (MLM) - BERT style
# Predict masked tokens given context
# P(x_masked | x_unmasked)

def create_mlm_data(tokens, mask_prob=0.15, vocab_size=30000):
    """Create MLM training data"""
    labels = tokens.clone()
    
    # Random mask positions
    mask = torch.rand(tokens.shape) < mask_prob
    
    # Don't mask special tokens
    special_tokens_mask = (tokens == 0) | (tokens == 1) | (tokens == 2)
    mask = mask & ~special_tokens_mask
    
    # 80% [MASK], 10% random, 10% unchanged
    indices_replaced = mask & (torch.rand(tokens.shape) < 0.8)
    tokens[indices_replaced] = 103  # [MASK] token
    
    indices_random = mask & ~indices_replaced & (torch.rand(tokens.shape) < 0.5)
    tokens[indices_random] = torch.randint(vocab_size, tokens.shape)[indices_random]
    
    # Labels: -100 for non-masked (ignored in loss)
    labels[~mask] = -100
    
    return tokens, labels


# 3. Span Corruption - T5 style
# Replace spans with sentinel tokens
```

### Scaling Laws

```
Chinchilla Scaling Laws:
- Optimal model size and data scale together
- N (params) ≈ D (tokens) for compute-optimal training
- Loss ∝ N^(-0.5) ∝ D^(-0.5) ∝ C^(-0.5)

Key insights:
- LLaMA 7B trained on 1T tokens (more data than Chinchilla suggests)
- Inference-optimal: smaller models trained longer
- Compute-optimal: balance model size and data
```

### Distributed Training

```python
# Data Parallelism
# - Replicate model across GPUs
# - Split batch across GPUs
# - Synchronize gradients

# Model Parallelism (Tensor Parallelism)
# - Split layers across GPUs
# - Each GPU holds part of each layer

# Pipeline Parallelism
# - Split layers across GPUs sequentially
# - Each GPU processes different micro-batches

# ZeRO (Zero Redundancy Optimizer)
# Stage 1: Partition optimizer states
# Stage 2: + Partition gradients
# Stage 3: + Partition parameters

# Example with PyTorch DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
model = DDP(model, device_ids=[local_rank])

# Example with DeepSpeed
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config={
        "train_batch_size": 32,
        "gradient_accumulation_steps": 4,
        "fp16": {"enabled": True},
        "zero_optimization": {"stage": 2}
    }
)
```

---

## 4. Fine-tuning Techniques

### Full Fine-tuning

```python
# Update all model parameters
model = AutoModelForCausalLM.from_pretrained("gpt2")

for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
```

### LoRA (Low-Rank Adaptation)

```python
class LoRALayer(nn.Module):
    """
    LoRA: W' = W + BA where B ∈ R^{d×r}, A ∈ R^{r×k}
    r << min(d, k) - low rank
    """
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False
        
        # Low-rank matrices
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize A with Gaussian, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.scaling = alpha / rank
    
    def forward(self, x):
        # Original output + LoRA adaptation
        original_output = self.original(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_output + lora_output


# Using PEFT library
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                      # Rank
    lora_alpha=32,            # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
```

### QLoRA (Quantized LoRA)

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True       # Nested quantization
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Memory: 7B model fits in ~6GB VRAM
```

### RLHF (Reinforcement Learning from Human Feedback)

```python
# Step 1: Supervised Fine-Tuning (SFT)
# Train on high-quality demonstration data

# Step 2: Train Reward Model
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.head = nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.head(last_hidden)
        return reward

# Training: Given (prompt, chosen, rejected) pairs
# Loss = -log(sigmoid(r_chosen - r_rejected))

# Step 3: PPO Training
# Maximize: E[R(x,y) - β*KL(π_θ || π_ref)]

# Using TRL library
from trl import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    batch_size=256,
    mini_batch_size=16,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)
```

### DPO (Direct Preference Optimization)

```python
# Simpler alternative to RLHF - no reward model needed
# Directly optimize policy from preferences

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta=0.1):
    """
    DPO Loss = -log(sigmoid(β * (log π(y_w|x) - log π(y_l|x) 
                                 - log π_ref(y_w|x) + log π_ref(y_l|x))))
    """
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps
    
    logits = beta * (policy_logratios - reference_logratios)
    loss = -F.logsigmoid(logits).mean()
    
    return loss

# Using TRL
from trl import DPOTrainer

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    beta=0.1,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
```

---

## 5. Prompt Engineering

### Prompting Techniques

```python
# 1. Zero-shot Prompting
prompt = """Classify the sentiment of the following text as positive, negative, or neutral.

Text: "The movie was absolutely fantastic!"
Sentiment:"""

# 2. Few-shot Prompting (In-Context Learning)
prompt = """Classify the sentiment:

Text: "I love this product!"
Sentiment: positive

Text: "This is terrible."
Sentiment: negative

Text: "The movie was absolutely fantastic!"
Sentiment:"""

# 3. Chain-of-Thought (CoT) Prompting
prompt = """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
Each can has 3 tennis balls. How many tennis balls does he have now?

A: Let's think step by step.
Roger started with 5 balls.
He bought 2 cans with 3 balls each = 2 × 3 = 6 balls.
Total = 5 + 6 = 11 balls.
The answer is 11.

Q: The cafeteria had 23 apples. They used 20 to make lunch and bought 6 more. 
How many apples do they have?

A: Let's think step by step."""

# 4. Self-Consistency (Sample multiple CoT paths, take majority vote)
responses = [generate_with_cot(prompt) for _ in range(5)]
final_answer = majority_vote(responses)

# 5. ReAct (Reasoning + Acting)
prompt = """Question: What is the capital of the country where the Eiffel Tower is located?

Thought: I need to find out which country the Eiffel Tower is in.
Action: Search[Eiffel Tower location]
Observation: The Eiffel Tower is located in Paris, France.
Thought: France is the country. Now I need to find its capital.
Action: Search[capital of France]
Observation: The capital of France is Paris.
Thought: I have the answer.
Action: Finish[Paris]"""
```

### Prompt Templates

```python
# System prompts
SYSTEM_PROMPT = """You are a helpful AI assistant. You provide accurate, 
concise responses. If you're unsure, say so. Never make up information."""

# Task-specific templates
CLASSIFICATION_TEMPLATE = """Classify the following text into one of these categories: {categories}

Text: {text}

Category:"""

SUMMARIZATION_TEMPLATE = """Summarize the following text in {num_sentences} sentences:

{text}

Summary:"""

QA_TEMPLATE = """Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:"""

# Structured output
JSON_TEMPLATE = """Extract the following information from the text and return as JSON:
- name: person's name
- age: person's age
- occupation: person's job

Text: {text}

JSON:"""
```

---

## 6. RAG (Retrieval-Augmented Generation)

### RAG Architecture

```python
# 1. Document Processing
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    return chunks


# 2. Embedding & Indexing
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorStore:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
    
    def add_documents(self, documents):
        embeddings = self.model.encode([doc.page_content for doc in documents])
        
        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim with normalized vectors)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.documents.extend(documents)
    
    def search(self, query, k=5):
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            results.append({
                "document": self.documents[idx],
                "score": float(score)
            })
        return results


# 3. RAG Pipeline
class RAGPipeline:
    def __init__(self, vector_store, llm, k=5):
        self.vector_store = vector_store
        self.llm = llm
        self.k = k
    
    def query(self, question):
        # Retrieve relevant documents
        results = self.vector_store.search(question, k=self.k)
        
        # Build context
        context = "\n\n".join([r["document"].page_content for r in results])
        
        # Generate answer
        prompt = f"""Answer the question based on the following context:

Context:
{context}

Question: {question}

Answer:"""
        
        answer = self.llm.generate(prompt)
        
        return {
            "answer": answer,
            "sources": results
        }
```

### Advanced RAG Techniques

```python
# 1. Hybrid Search (Dense + Sparse)
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, documents, embedding_model):
        self.documents = documents
        self.texts = [doc.page_content for doc in documents]
        
        # Dense retriever
        self.embedder = SentenceTransformer(embedding_model)
        self.embeddings = self.embedder.encode(self.texts)
        
        # Sparse retriever (BM25)
        tokenized = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query, k=5, alpha=0.5):
        # Dense scores
        query_emb = self.embedder.encode([query])
        dense_scores = np.dot(self.embeddings, query_emb.T).flatten()
        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
        
        # Sparse scores
        sparse_scores = self.bm25.get_scores(query.lower().split())
        sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min())
        
        # Combine
        combined = alpha * dense_scores + (1 - alpha) * sparse_scores
        
        top_k = np.argsort(combined)[-k:][::-1]
        return [self.documents[i] for i in top_k]


# 2. Query Expansion/Rewriting
def expand_query(query, llm):
    prompt = f"""Generate 3 alternative versions of this search query 
that might help find relevant documents:

Original query: {query}

Alternative queries:"""
    
    alternatives = llm.generate(prompt)
    return [query] + alternatives.split("\n")


# 3. Re-ranking
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, documents, top_k=5):
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)
        
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [documents[i] for i in ranked_indices]


# 4. Contextual Compression
def compress_context(query, documents, llm):
    """Extract only relevant parts of documents"""
    compressed = []
    for doc in documents:
        prompt = f"""Extract only the parts of the following text that are 
relevant to answering: "{query}"

Text: {doc.page_content}

Relevant parts:"""
        
        relevant = llm.generate(prompt)
        if relevant.strip():
            compressed.append(relevant)
    return compressed
```

### Vector Databases

```python
# Pinecone
import pinecone

pinecone.init(api_key="your-api-key", environment="us-west1-gcp")
index = pinecone.Index("my-index")

index.upsert(vectors=[
    {"id": "doc1", "values": embedding1, "metadata": {"text": "..."}},
    {"id": "doc2", "values": embedding2, "metadata": {"text": "..."}},
])

results = index.query(vector=query_embedding, top_k=5, include_metadata=True)


# Chroma
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_collection")

collection.add(
    documents=["doc1", "doc2"],
    metadatas=[{"source": "a"}, {"source": "b"}],
    ids=["id1", "id2"]
)

results = collection.query(query_texts=["query"], n_results=5)


# Weaviate
import weaviate

client = weaviate.Client("http://localhost:8080")

client.schema.create_class({
    "class": "Document",
    "vectorizer": "text2vec-transformers",
})

client.data_object.create(
    {"content": "document text"},
    "Document"
)
```

---

## 7. LLM Agents & Tools

### Agent Architecture

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.tools import BaseTool

# Define tools
class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Useful for math calculations. Input should be a math expression."
    
    def _run(self, query: str) -> str:
        try:
            return str(eval(query))
        except:
            return "Error: Invalid expression"

class SearchTool(BaseTool):
    name = "search"
    description = "Search the web for information. Input should be a search query."
    
    def _run(self, query: str) -> str:
        # Implement search logic
        return "Search results..."


# ReAct Agent
REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


class ReActAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = 10
    
    def run(self, question):
        scratchpad = ""
        
        for i in range(self.max_iterations):
            prompt = REACT_PROMPT.format(
                tools="\n".join([f"{t.name}: {t.description}" for t in self.tools.values()]),
                tool_names=", ".join(self.tools.keys()),
                input=question,
                agent_scratchpad=scratchpad
            )
            
            response = self.llm.generate(prompt, stop=["Observation:"])
            
            # Parse response
            if "Final Answer:" in response:
                return response.split("Final Answer:")[-1].strip()
            
            # Extract action and input
            action_match = re.search(r"Action: (\w+)", response)
            input_match = re.search(r"Action Input: (.+)", response)
            
            if action_match and input_match:
                action = action_match.group(1)
                action_input = input_match.group(1)
                
                # Execute tool
                if action in self.tools:
                    observation = self.tools[action]._run(action_input)
                else:
                    observation = f"Unknown tool: {action}"
                
                scratchpad += response + f"\nObservation: {observation}\nThought:"
        
        return "Max iterations reached"
```

### Function Calling

```python
# OpenAI Function Calling
import openai

functions = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    }
]

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    functions=functions,
    function_call="auto"
)

# Parse function call
if response.choices[0].message.get("function_call"):
    func_name = response.choices[0].message.function_call.name
    func_args = json.loads(response.choices[0].message.function_call.arguments)
    
    # Execute function
    result = execute_function(func_name, func_args)
```

---

## 8. LLM Evaluation

### Evaluation Metrics

```python
# 1. Perplexity (language modeling quality)
def perplexity(logits, labels):
    """Lower is better"""
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    return torch.exp(loss)


# 2. BLEU Score (translation/generation quality)
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

reference = [["the", "cat", "sat", "on", "mat"]]
candidate = ["the", "cat", "is", "on", "mat"]
score = sentence_bleu(reference, candidate)


# 3. ROUGE Score (summarization)
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
scores = scorer.score("reference text", "generated text")


# 4. BERTScore (semantic similarity)
from bert_score import score

P, R, F1 = score(candidates, references, lang="en")


# 5. LLM-as-Judge
def llm_judge(question, answer, reference=None):
    prompt = f"""Rate the following answer on a scale of 1-5 for:
- Accuracy: Is the information correct?
- Relevance: Does it answer the question?
- Completeness: Is it thorough?
- Clarity: Is it well-written?

Question: {question}
Answer: {answer}
{"Reference: " + reference if reference else ""}

Provide ratings and brief justification for each criterion.
"""
    return llm.generate(prompt)
```

### RAG Evaluation (RAGAS)

```python
# RAGAS metrics
# pip install ragas

from ragas.metrics import (
    faithfulness,      # Is answer grounded in context?
    answer_relevancy,  # Is answer relevant to question?
    context_precision, # Is retrieved context relevant?
    context_recall     # Does context contain answer?
)

from ragas import evaluate

results = evaluate(
    dataset,  # Contains question, answer, contexts, ground_truth
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
```

---

## 9. LLM Inference Optimization

### KV Cache

```python
class KVCache:
    """Cache key-value pairs to avoid recomputation"""
    def __init__(self):
        self.cache = {}
    
    def update(self, layer_idx, key, value):
        if layer_idx not in self.cache:
            self.cache[layer_idx] = {"key": key, "value": value}
        else:
            self.cache[layer_idx]["key"] = torch.cat([
                self.cache[layer_idx]["key"], key
            ], dim=2)
            self.cache[layer_idx]["value"] = torch.cat([
                self.cache[layer_idx]["value"], value
            ], dim=2)
    
    def get(self, layer_idx):
        return self.cache.get(layer_idx, {"key": None, "value": None})

# During generation, only compute attention for new token
# Keys and values for previous tokens are cached
```

### Quantization

```python
# INT8 Quantization
import torch.quantization as quant

# Dynamic quantization (weights quantized, activations computed in FP32)
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Static quantization (both weights and activations quantized)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model)
# Calibrate with representative data
model_int8 = torch.quantization.convert(model_prepared)

# GPTQ (post-training quantization for LLMs)
# Quantizes weights to 4-bit with minimal accuracy loss

# AWQ (Activation-aware Weight Quantization)
# Protects important weights based on activation magnitudes
```

### Speculative Decoding

```python
def speculative_decode(target_model, draft_model, prompt, gamma=4):
    """
    Use small draft model to propose tokens, verify with large model
    - gamma: number of draft tokens per step
    """
    generated = prompt
    
    while not done:
        # Draft model generates gamma tokens
        draft_tokens = []
        for _ in range(gamma):
            draft_token = draft_model.generate_one(generated + draft_tokens)
            draft_tokens.append(draft_token)
        
        # Target model verifies all at once
        target_logits = target_model(generated + draft_tokens)
        
        # Accept/reject each draft token
        accepted = 0
        for i, draft_token in enumerate(draft_tokens):
            target_prob = softmax(target_logits[i])[draft_token]
            draft_prob = softmax(draft_model(generated + draft_tokens[:i]))[draft_token]
            
            if random() < min(1, target_prob / draft_prob):
                generated.append(draft_token)
                accepted += 1
            else:
                # Sample from adjusted distribution
                new_token = sample_adjusted(target_logits[i], draft_logits[i])
                generated.append(new_token)
                break
    
    return generated

# Speed up: 2-3x faster inference with same output distribution
```

### Batching Strategies

```python
# Continuous Batching (Dynamic Batching)
# - Add new requests to batch as old ones complete
# - No waiting for entire batch to finish

class ContinuousBatcher:
    def __init__(self, max_batch_size=32):
        self.max_batch_size = max_batch_size
        self.active_requests = []
        self.waiting_queue = []
    
    def step(self):
        # Add waiting requests to batch
        while len(self.active_requests) < self.max_batch_size and self.waiting_queue:
            self.active_requests.append(self.waiting_queue.pop(0))
        
        # Run one generation step for all active requests
        outputs = model.generate_step(self.active_requests)
        
        # Remove completed requests
        completed = []
        for req, output in zip(self.active_requests, outputs):
            if output.is_done:
                completed.append(req)
        
        for req in completed:
            self.active_requests.remove(req)
            yield req.result


# PagedAttention (vLLM)
# - Manage KV cache like virtual memory pages
# - Enables efficient memory sharing and batching
```

---

## 10. Production Deployment

### Model Serving

```python
# FastAPI endpoint
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/generate")
async def generate(request: GenerateRequest):
    output = model.generate(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    return {"generated_text": output}


# vLLM serving
# pip install vllm
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

outputs = llm.generate(["Hello, my name is"], sampling_params)


# TensorRT-LLM (NVIDIA)
# Optimized inference with TensorRT
# Supports INT8, FP8, and various optimizations
```

### Guardrails & Safety

```python
# Content filtering
class ContentFilter:
    def __init__(self):
        self.blocked_patterns = [...]
        self.classifier = load_safety_classifier()
    
    def filter_input(self, text):
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if pattern in text.lower():
                return False, "Blocked content detected"
        return True, None
    
    def filter_output(self, text):
        # Classify output safety
        score = self.classifier.predict(text)
        if score < 0.5:
            return False, "Potentially harmful content"
        return True, None


# Prompt injection detection
def detect_injection(user_input):
    injection_patterns = [
        "ignore previous instructions",
        "disregard all prior",
        "you are now",
        "pretend you are"
    ]
    
    for pattern in injection_patterns:
        if pattern in user_input.lower():
            return True
    return False


# Rate limiting
from functools import wraps
import time

class RateLimiter:
    def __init__(self, max_requests, window_seconds):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = {}
    
    def is_allowed(self, user_id):
        now = time.time()
        user_requests = self.requests.get(user_id, [])
        
        # Remove old requests
        user_requests = [t for t in user_requests if now - t < self.window]
        
        if len(user_requests) >= self.max_requests:
            return False
        
        user_requests.append(now)
        self.requests[user_id] = user_requests
        return True
```

### Monitoring & Observability

```python
# Key metrics to track:
# - Latency (p50, p95, p99)
# - Throughput (requests/second)
# - Token usage
# - Error rates
# - Cache hit rates
# - GPU utilization

import prometheus_client as prom

REQUEST_LATENCY = prom.Histogram('request_latency_seconds', 'Request latency')
TOKEN_COUNT = prom.Counter('tokens_generated_total', 'Total tokens generated')
ERROR_COUNT = prom.Counter('errors_total', 'Total errors', ['type'])

@REQUEST_LATENCY.time()
def generate(prompt):
    try:
        output = model.generate(prompt)
        TOKEN_COUNT.inc(len(output.tokens))
        return output
    except Exception as e:
        ERROR_COUNT.labels(type=type(e).__name__).inc()
        raise
```

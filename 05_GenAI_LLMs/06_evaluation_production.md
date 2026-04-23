# LLM Evaluation & Production - Complete Guide

## Table of Contents
1. [Evaluation Metrics](#evaluation-metrics)
2. [Benchmarks and Datasets](#benchmarks-and-datasets)
3. [LLM-as-Judge](#llm-as-judge)
4. [Inference Optimization](#inference-optimization)
5. [Production Deployment](#production-deployment)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Interview Questions](#interview-questions)

---

## 1. Evaluation Metrics

### Language Modeling Metrics

```python
import torch
import torch.nn.functional as F
import numpy as np

def perplexity(logits, labels, ignore_index=-100):
    """
    Perplexity: Measures how well model predicts next token.
    Lower is better. PPL = exp(cross_entropy_loss)
    
    Interpretation:
    - PPL=1: Perfect prediction
    - PPL=V: Random guess among V tokens
    """
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
        reduction='mean'
    )
    return torch.exp(loss).item()


def bits_per_character(loss, tokenizer_compression_ratio=4.0):
    """
    BPC: Bits per character. More interpretable across tokenizers.
    """
    bits_per_token = loss / np.log(2)
    return bits_per_token / tokenizer_compression_ratio
```

### Generation Quality Metrics

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge_score import rouge_scorer

# BLEU Score (translation, summarization)
def calculate_bleu(reference: str, candidate: str, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    BLEU: Bilingual Evaluation Understudy
    Measures n-gram overlap between candidate and reference.
    
    weights: (1-gram, 2-gram, 3-gram, 4-gram) weights
    """
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    
    return sentence_bleu(reference_tokens, candidate_tokens, weights=weights)


# ROUGE Score (summarization)
def calculate_rouge(reference: str, candidate: str):
    """
    ROUGE: Recall-Oriented Understudy for Gisting Evaluation
    
    ROUGE-1: Unigram overlap
    ROUGE-2: Bigram overlap  
    ROUGE-L: Longest common subsequence
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    scores = scorer.score(reference, candidate)
    
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


# BERTScore (semantic similarity)
def calculate_bertscore(references, candidates, lang='en'):
    """
    BERTScore: Uses BERT embeddings for semantic similarity.
    More robust than n-gram based metrics.
    """
    from bert_score import score
    
    P, R, F1 = score(candidates, references, lang=lang)
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }


# METEOR (machine translation)
def calculate_meteor(reference: str, candidate: str):
    """
    METEOR: Considers synonyms, stemming, and paraphrases.
    """
    from nltk.translate.meteor_score import meteor_score
    return meteor_score([reference.split()], candidate.split())
```

### Task-Specific Metrics

```python
# Exact Match (QA)
def exact_match(prediction: str, ground_truth: str) -> float:
    """Strict string matching (normalized)."""
    def normalize(s):
        return s.lower().strip()
    return float(normalize(prediction) == normalize(ground_truth))


# F1 Score (QA, token-level)
def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 for extractive QA."""
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())
    
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    
    common = pred_tokens & truth_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# Pass@k (Code Generation)
def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Pass@k: Probability that at least one of k samples passes.
    
    n: total samples generated
    c: number that pass tests
    k: number of samples to consider
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

---

## 2. Benchmarks and Datasets

### Common LLM Benchmarks

```python
"""
Major Benchmarks:

1. MMLU (Massive Multitask Language Understanding)
   - 57 subjects: STEM, humanities, social sciences
   - Multiple choice
   - Tests knowledge and reasoning

2. HellaSwag
   - Commonsense reasoning
   - Sentence completion
   - Tests world knowledge

3. ARC (AI2 Reasoning Challenge)
   - Science questions
   - Easy and Challenge sets
   - Multiple choice

4. WinoGrande
   - Coreference resolution
   - Commonsense reasoning
   - Fill-in-the-blank

5. GSM8K
   - Grade school math
   - Multi-step reasoning
   - Tests arithmetic and word problems

6. HumanEval / MBPP
   - Code generation
   - Function completion
   - Pass@k evaluation

7. TruthfulQA
   - Factual accuracy
   - Tests for common misconceptions
   - Measures hallucination

8. MT-Bench
   - Multi-turn conversation
   - Various domains
   - LLM-as-judge evaluation
"""

# Running evaluations with lm-evaluation-harness
"""
pip install lm-eval

# Run evaluation
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks mmlu,hellaswag,arc_easy,arc_challenge \
    --batch_size 8
"""

# Custom evaluation dataset
from datasets import Dataset

def create_eval_dataset(examples):
    """Create evaluation dataset."""
    return Dataset.from_dict({
        'question': [e['question'] for e in examples],
        'answer': [e['answer'] for e in examples],
        'context': [e.get('context', '') for e in examples]
    })
```

### Evaluation Harness

```python
from typing import List, Dict, Callable

class EvaluationHarness:
    """Framework for running LLM evaluations."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def evaluate_multiple_choice(self, dataset, prompt_template: str) -> Dict:
        """Evaluate on multiple choice questions."""
        correct = 0
        total = 0
        
        for example in dataset:
            prompt = prompt_template.format(**example)
            
            # Get log probabilities for each choice
            choices = example['choices']
            choice_probs = []
            
            for choice in choices:
                full_prompt = prompt + choice
                tokens = self.tokenizer.encode(full_prompt, return_tensors='pt')
                
                with torch.no_grad():
                    outputs = self.model(tokens)
                    logits = outputs.logits[0, -1]
                    prob = F.softmax(logits, dim=-1)
                    
                    # Get probability of choice token
                    choice_token = self.tokenizer.encode(choice)[0]
                    choice_probs.append(prob[choice_token].item())
            
            predicted = choices[np.argmax(choice_probs)]
            correct += int(predicted == example['answer'])
            total += 1
        
        return {
            'accuracy': correct / total,
            'correct': correct,
            'total': total
        }
    
    def evaluate_generation(self, dataset, prompt_template: str,
                           metrics: List[Callable]) -> Dict:
        """Evaluate on generation tasks."""
        results = {metric.__name__: [] for metric in metrics}
        
        for example in dataset:
            prompt = prompt_template.format(**example)
            
            # Generate
            inputs = self.tokenizer(prompt, return_tensors='pt')
            outputs = self.model.generate(**inputs, max_new_tokens=256)
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from output
            generated = generated[len(prompt):].strip()
            
            # Calculate metrics
            for metric in metrics:
                score = metric(generated, example['answer'])
                results[metric.__name__].append(score)
        
        return {
            name: np.mean(scores) for name, scores in results.items()
        }
```

---

## 3. LLM-as-Judge

### Pairwise Comparison

```python
PAIRWISE_TEMPLATE = """You are an impartial judge comparing two AI assistant responses.

User Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Evaluate both responses on:
1. Helpfulness: Does it address the user's question?
2. Accuracy: Is the information correct?
3. Clarity: Is it well-written and easy to understand?

Which response is better? Answer with:
- "A" if Response A is better
- "B" if Response B is better
- "TIE" if they are equally good

Provide brief reasoning, then give your verdict.

Reasoning:"""


class PairwiseJudge:
    """LLM-based pairwise comparison."""
    
    def __init__(self, judge_model):
        self.judge = judge_model
    
    def compare(self, question: str, response_a: str, response_b: str) -> Dict:
        prompt = PAIRWISE_TEMPLATE.format(
            question=question,
            response_a=response_a,
            response_b=response_b
        )
        
        judgment = self.judge.generate(prompt, max_tokens=500)
        
        # Parse verdict
        if "Verdict: A" in judgment or judgment.strip().endswith("A"):
            winner = "A"
        elif "Verdict: B" in judgment or judgment.strip().endswith("B"):
            winner = "B"
        else:
            winner = "TIE"
        
        return {
            'winner': winner,
            'reasoning': judgment
        }
    
    def evaluate_model(self, model_a, model_b, test_set) -> Dict:
        """Run pairwise evaluation on test set."""
        wins_a = 0
        wins_b = 0
        ties = 0
        
        for example in test_set:
            # Generate responses
            response_a = model_a.generate(example['question'])
            response_b = model_b.generate(example['question'])
            
            # Compare (with position debiasing)
            result1 = self.compare(example['question'], response_a, response_b)
            result2 = self.compare(example['question'], response_b, response_a)
            
            # Consistent verdict
            if result1['winner'] == 'A' and result2['winner'] == 'B':
                wins_a += 1
            elif result1['winner'] == 'B' and result2['winner'] == 'A':
                wins_b += 1
            else:
                ties += 1
        
        total = len(test_set)
        return {
            'model_a_win_rate': wins_a / total,
            'model_b_win_rate': wins_b / total,
            'tie_rate': ties / total
        }
```

### Single-Response Scoring

```python
SCORING_TEMPLATE = """Rate the following AI assistant response on a scale of 1-10.

User Question: {question}

AI Response:
{response}

Evaluate on these criteria:
1. Helpfulness (1-10): Does it address the question?
2. Accuracy (1-10): Is the information correct?
3. Clarity (1-10): Is it well-written?
4. Safety (1-10): Is it appropriate and harmless?

Provide scores for each criterion and an overall score.

Scores:"""


class ResponseScorer:
    """Score individual responses."""
    
    def __init__(self, judge_model):
        self.judge = judge_model
    
    def score(self, question: str, response: str) -> Dict:
        prompt = SCORING_TEMPLATE.format(
            question=question,
            response=response
        )
        
        judgment = self.judge.generate(prompt, max_tokens=300)
        
        # Parse scores
        scores = self._parse_scores(judgment)
        return scores
    
    def _parse_scores(self, judgment: str) -> Dict:
        """Extract numerical scores from judgment."""
        import re
        
        scores = {}
        patterns = {
            'helpfulness': r'[Hh]elpfulness[:\s]*(\d+)',
            'accuracy': r'[Aa]ccuracy[:\s]*(\d+)',
            'clarity': r'[Cc]larity[:\s]*(\d+)',
            'safety': r'[Ss]afety[:\s]*(\d+)',
            'overall': r'[Oo]verall[:\s]*(\d+)'
        }
        
        for name, pattern in patterns.items():
            match = re.search(pattern, judgment)
            if match:
                scores[name] = int(match.group(1))
        
        return scores
```

---

## 4. Inference Optimization

### Quantization

```python
"""
Quantization: Reduce model precision to save memory and speed up inference.

Types:
- INT8: 8-bit integers (4x smaller than FP32)
- INT4: 4-bit integers (8x smaller than FP32)
- NF4: NormalFloat4 (better for normally distributed weights)
- GPTQ: Post-training quantization
- AWQ: Activation-aware weight quantization
"""

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8-bit quantization
model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)

# 4-bit quantization (QLoRA style)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# GPTQ quantization
"""
pip install auto-gptq

from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    device="cuda:0"
)
"""
```

### KV Cache Optimization

```python
"""
KV Cache: Store computed key-value pairs for efficient generation.

Optimizations:
1. Paged Attention (vLLM): Efficient memory management
2. Flash Attention: Reduced memory bandwidth
3. Multi-Query Attention: Shared KV heads
4. Sliding Window: Limited context caching
"""

# vLLM for efficient serving
"""
pip install vllm

from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,  # Multi-GPU
    gpu_memory_utilization=0.9
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)

outputs = llm.generate(["Hello, world!"], sampling_params)
"""
```

### Batching and Streaming

```python
from typing import List, Iterator
import asyncio

class BatchedInference:
    """Batch multiple requests for efficiency."""
    
    def __init__(self, model, tokenizer, max_batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for batch of prompts."""
        # Tokenize with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)
        
        # Generate
        outputs = self.model.generate(**inputs, **kwargs)
        
        # Decode
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Remove prompts
        return [r[len(p):] for r, p in zip(responses, prompts)]


class StreamingGenerator:
    """Stream tokens as they're generated."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_stream(self, prompt: str, max_tokens: int = 256) -> Iterator[str]:
        """Yield tokens as they're generated."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        generated = inputs.input_ids
        
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(generated)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=-1)
                
                token_str = self.tokenizer.decode(next_token[0])
                yield token_str
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
```

---

## 5. Production Deployment

### Model Serving Architecture

```python
"""
Production serving options:

1. vLLM: High-throughput serving with PagedAttention
2. TGI (Text Generation Inference): HuggingFace's solution
3. TensorRT-LLM: NVIDIA optimized inference
4. Triton Inference Server: Multi-model serving

Architecture patterns:
- Single model API
- Model router (different models for different tasks)
- Load balancer for horizontal scaling
- GPU cluster management
"""

# FastAPI wrapper for model serving
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    latency_ms: float

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    import time
    start = time.time()
    
    try:
        output = model.generate(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        latency = (time.time() - start) * 1000
        
        return GenerateResponse(
            text=output,
            tokens_generated=len(tokenizer.encode(output)),
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### Scaling and Load Balancing

```python
"""
Scaling strategies:

1. Vertical scaling: Bigger GPU, more VRAM
2. Horizontal scaling: Multiple instances
3. Model parallelism: Split across GPUs
4. Pipeline parallelism: Different layers on different GPUs

Load balancing:
- Round-robin: Simple, equal distribution
- Least connections: Route to least busy
- Weighted: Route based on GPU capability
- Smart routing: Route based on request complexity
"""

import asyncio
from typing import List
import random

class LoadBalancer:
    """Simple load balancer for model instances."""
    
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.health_status = {ep: True for ep in endpoints}
        self.request_counts = {ep: 0 for ep in endpoints}
    
    def get_endpoint(self, strategy: str = "least_connections") -> str:
        """Select endpoint based on strategy."""
        healthy = [ep for ep in self.endpoints if self.health_status[ep]]
        
        if not healthy:
            raise Exception("No healthy endpoints available")
        
        if strategy == "round_robin":
            return random.choice(healthy)
        
        elif strategy == "least_connections":
            return min(healthy, key=lambda ep: self.request_counts[ep])
        
        return healthy[0]
    
    async def forward_request(self, request, strategy="least_connections"):
        """Forward request to selected endpoint."""
        endpoint = self.get_endpoint(strategy)
        self.request_counts[endpoint] += 1
        
        try:
            response = await self._send_request(endpoint, request)
            return response
        finally:
            self.request_counts[endpoint] -= 1
    
    async def health_check(self):
        """Periodic health check of all endpoints."""
        while True:
            for endpoint in self.endpoints:
                try:
                    # Simple health check
                    response = await self._send_request(endpoint + "/health", {})
                    self.health_status[endpoint] = response.get("status") == "healthy"
                except:
                    self.health_status[endpoint] = False
            
            await asyncio.sleep(30)  # Check every 30 seconds
```

---

## 6. Monitoring and Observability

### Metrics Collection

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
import time

@dataclass
class RequestMetrics:
    request_id: str
    timestamp: datetime
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str
    status: str
    error: str = None

class MetricsCollector:
    """Collect and aggregate LLM metrics."""
    
    def __init__(self):
        self.metrics: List[RequestMetrics] = []
    
    def record(self, metrics: RequestMetrics):
        """Record a request's metrics."""
        self.metrics.append(metrics)
    
    def get_summary(self, window_minutes: int = 60) -> Dict:
        """Get summary statistics for time window."""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent = [m for m in self.metrics if m.timestamp > cutoff]
        
        if not recent:
            return {}
        
        return {
            'total_requests': len(recent),
            'success_rate': sum(1 for m in recent if m.status == 'success') / len(recent),
            'avg_latency_ms': sum(m.latency_ms for m in recent) / len(recent),
            'p50_latency_ms': np.percentile([m.latency_ms for m in recent], 50),
            'p95_latency_ms': np.percentile([m.latency_ms for m in recent], 95),
            'p99_latency_ms': np.percentile([m.latency_ms for m in recent], 99),
            'total_prompt_tokens': sum(m.prompt_tokens for m in recent),
            'total_completion_tokens': sum(m.completion_tokens for m in recent),
            'tokens_per_second': sum(m.completion_tokens for m in recent) / (window_minutes * 60)
        }


# Prometheus metrics
"""
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('llm_requests_total', 'Total LLM requests', ['model', 'status'])
REQUEST_LATENCY = Histogram('llm_request_latency_seconds', 'Request latency', ['model'])
TOKENS_GENERATED = Counter('llm_tokens_generated_total', 'Tokens generated', ['model'])
ACTIVE_REQUESTS = Gauge('llm_active_requests', 'Currently processing requests')
"""
```

### Logging and Tracing

```python
import logging
import json
from uuid import uuid4

class LLMLogger:
    """Structured logging for LLM requests."""
    
    def __init__(self, logger_name: str = "llm"):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
    
    def log_request(self, request_id: str, prompt: str, params: Dict):
        """Log incoming request."""
        self.logger.info(json.dumps({
            'event': 'request',
            'request_id': request_id,
            'prompt_length': len(prompt),
            'params': params,
            'timestamp': datetime.now().isoformat()
        }))
    
    def log_response(self, request_id: str, response: str, 
                     latency_ms: float, tokens: int):
        """Log response."""
        self.logger.info(json.dumps({
            'event': 'response',
            'request_id': request_id,
            'response_length': len(response),
            'latency_ms': latency_ms,
            'tokens_generated': tokens,
            'timestamp': datetime.now().isoformat()
        }))
    
    def log_error(self, request_id: str, error: str):
        """Log error."""
        self.logger.error(json.dumps({
            'event': 'error',
            'request_id': request_id,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }))


class RequestTracer:
    """Distributed tracing for LLM pipelines."""
    
    def __init__(self):
        self.traces = {}
    
    def start_trace(self, request_id: str = None) -> str:
        """Start a new trace."""
        trace_id = request_id or str(uuid4())
        self.traces[trace_id] = {
            'start_time': time.time(),
            'spans': []
        }
        return trace_id
    
    def add_span(self, trace_id: str, name: str, metadata: Dict = None):
        """Add a span to the trace."""
        if trace_id in self.traces:
            self.traces[trace_id]['spans'].append({
                'name': name,
                'timestamp': time.time(),
                'metadata': metadata or {}
            })
    
    def end_trace(self, trace_id: str) -> Dict:
        """End trace and return data."""
        if trace_id in self.traces:
            trace = self.traces[trace_id]
            trace['end_time'] = time.time()
            trace['duration_ms'] = (trace['end_time'] - trace['start_time']) * 1000
            return self.traces.pop(trace_id)
        return None
```

---

## 7. Interview Questions

```python
# Q1: How do you evaluate LLM quality?
"""
Evaluation approaches:

1. Automated metrics
   - Perplexity (language modeling)
   - BLEU, ROUGE (generation)
   - Pass@k (code)
   - Exact match, F1 (QA)

2. Benchmarks
   - MMLU (knowledge)
   - HumanEval (coding)
   - GSM8K (math)
   - TruthfulQA (factuality)

3. LLM-as-Judge
   - Pairwise comparison
   - Single response scoring
   - Reference-free evaluation

4. Human evaluation
   - A/B testing
   - Preference ratings
   - Task completion studies

Best practice: Combine multiple approaches
"""


# Q2: What is perplexity and what are its limitations?
"""
Perplexity = exp(cross_entropy_loss)

Interpretation:
- Lower is better
- PPL=1: Perfect prediction
- PPL=V: Random among V tokens

Limitations:
1. Tokenizer-dependent
   - Different tokenizers → different PPL
   - Can't compare across models with different tokenizers

2. Doesn't measure generation quality
   - Low PPL ≠ good responses
   - Doesn't capture factuality, coherence

3. Doesn't measure task performance
   - A model with higher PPL might be better at specific tasks

Better alternatives:
- Task-specific benchmarks
- Human evaluation
- LLM-as-judge
"""


# Q3: How does LLM-as-Judge work and what are the pitfalls?
"""
LLM-as-Judge: Use strong LLM to evaluate other LLMs

Approaches:
- Pairwise comparison (which is better?)
- Single scoring (rate 1-10)
- Reference-based (compare to gold answer)

Pitfalls:
1. Position bias
   - Prefers first/second option
   - Solution: Swap positions, average

2. Self-preference
   - Model prefers own outputs
   - Solution: Use different judge model

3. Verbosity bias
   - Longer = better
   - Solution: Include length in evaluation

4. Limited domain expertise
   - May not catch factual errors
   - Solution: Domain-specific judges

Best practices:
- Use position debiasing
- Multiple judges
- Calibrate on human judgments
"""


# Q4: How do you optimize LLM inference latency?
"""
Optimization techniques:

1. Quantization
   - INT8, INT4, NF4
   - 2-4x speedup, minimal quality loss

2. KV Cache
   - Store computed keys/values
   - Essential for generation

3. Batching
   - Process multiple requests together
   - Better GPU utilization

4. Flash Attention
   - Memory-efficient attention
   - 2-4x faster

5. Speculative decoding
   - Small model proposes, large verifies
   - Up to 2-3x speedup

6. Model parallelism
   - Split across GPUs
   - For large models

Tools: vLLM, TensorRT-LLM, TGI
"""


# Q5: What metrics should you monitor in production?
"""
Key metrics:

1. Latency
   - P50, P95, P99
   - Time to first token
   - Tokens per second

2. Throughput
   - Requests per second
   - Concurrent requests

3. Quality
   - Error rates
   - User feedback
   - Automated quality checks

4. Cost
   - Tokens consumed
   - GPU utilization
   - Cost per request

5. Reliability
   - Uptime
   - Error rates by type
   - Retry rates

Alerting thresholds:
- P99 latency > SLA
- Error rate > 1%
- GPU utilization > 90%
"""


# Q6: How do you handle model versioning and rollbacks?
"""
Model versioning strategies:

1. Immutable deployments
   - Each version gets new endpoint
   - /v1/generate, /v2/generate
   - Gradual migration

2. Blue-green deployment
   - Two production environments
   - Switch traffic atomically
   - Easy rollback

3. Canary releases
   - Route % of traffic to new model
   - Monitor metrics
   - Gradually increase

4. A/B testing
   - Split traffic for comparison
   - Statistical significance testing

Rollback triggers:
- Error rate spike
- Latency degradation
- Quality metric drop
- User feedback

Best practice: Automate rollback on metric thresholds
"""


# Q7: Compare different model serving frameworks
"""
vLLM:
- PagedAttention for memory efficiency
- High throughput
- Good for batch processing
- Python-native

TGI (Text Generation Inference):
- HuggingFace's solution
- Production-ready
- Good streaming support
- Docker-first

TensorRT-LLM:
- NVIDIA optimized
- Best raw performance
- More complex setup
- Requires TensorRT

Triton:
- Multi-model serving
- Model ensembles
- Production battle-tested
- Learning curve

Recommendation:
- Simple: TGI
- High throughput: vLLM
- Max performance: TensorRT-LLM
- Enterprise: Triton
"""


# Q8: How do you scale LLM serving horizontally?
"""
Horizontal scaling approach:

1. Stateless design
   - No session state in servers
   - KV cache managed per-request

2. Load balancing
   - Round-robin for uniform requests
   - Least-connections for varying load
   - Content-based for model routing

3. Auto-scaling
   - Based on queue depth
   - Based on GPU utilization
   - Based on latency targets

4. Request queuing
   - Buffer during spikes
   - Prioritization
   - Timeout handling

Challenges:
- Long-running requests
- KV cache memory
- Cold start times
- Uneven GPU utilization

Solutions:
- Request chunking
- Prewarming
- Smart scheduling
"""


# Q9: What is speculative decoding?
"""
Speculative decoding: Use small model to speed up large model

Process:
1. Small draft model generates K tokens quickly
2. Large model verifies all K tokens in parallel
3. Accept matching prefix, reject rest
4. Continue from first rejection

Why it works:
- Large model verification is parallel (single forward pass)
- Draft model is much faster
- Most draft tokens are accepted

Speedup: 2-3x typical, depends on acceptance rate

Requirements:
- Compatible tokenizers
- Draft model with similar distribution
- Overhead: draft generation + verification

Examples:
- LLaMA-70B with LLaMA-7B draft
- Custom distilled draft models
"""


# Q10: How do you ensure LLM output safety in production?
"""
Safety measures:

1. Input filtering
   - Block known attack patterns
   - Content classification
   - Rate limiting

2. Output filtering
   - Toxicity detection
   - PII detection
   - Fact checking

3. Guardrails
   - Topic restrictions
   - Format constraints
   - Length limits

4. Monitoring
   - Anomaly detection
   - Human review sampling
   - User reports

5. Model-level
   - RLHF alignment
   - Constitutional AI
   - Safety fine-tuning

Implementation:
- Pre-processing pipeline
- Post-processing checks
- Async review queue
- Feedback loops

Tools: NeMo Guardrails, Guardrails AI, custom classifiers
"""
```

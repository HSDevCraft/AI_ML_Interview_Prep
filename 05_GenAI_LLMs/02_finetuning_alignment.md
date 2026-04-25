# Fine-tuning & Alignment - Complete Guide

## ⚡ Interview Quick Summary

> **When asked about fine-tuning**: Always start with the decision framework, not the method. Ask: what do you want to change (knowledge vs behavior), how much compute/data do you have, and what's the latency budget?

### Fine-tuning Decision Tree

```
Do you want to change MODEL BEHAVIOR (style, format, safety)?
  YES → RLHF / DPO (preference learning)
  NO  → Continue...

Do you want to inject DOMAIN KNOWLEDGE?
  YES + knowledge is static → Full fine-tune or LoRA
  YES + knowledge changes   → RAG is better!
  NO  → Continue...

Do you have LARGE COMPUTE?
  YES + large dataset → Full fine-tuning
  NO  → LoRA (r=8-64) or QLoRA (consumer GPU)

Do you need MULTI-TASK switching?
  YES → LoRA adapters per task, swap at inference
  NO  → Merge LoRA into base model
```

### Method Comparison at a Glance

| Method | Trainable % | GPU Memory | Performance | Best For |
|--------|-------------|------------|-------------|----------|
| Full FT | 100% | Very High | Best | Large compute, domain shift |
| LoRA | 0.1-1% | Low | Near full | Most use cases |
| QLoRA | 0.1-1% | Very Low | Good | Consumer GPU (7B+ models) |
| Prefix-tuning | 0.1% | Low | Good | Soft prompts, generation |
| Adapter | 1-5% | Medium | Good | Modular, multi-task |

### 🚨 Top Interview Pitfalls
- Saying "fine-tune" when RAG would be better (don't inject rapidly-changing facts into weights)
- Not addressing **catastrophic forgetting** (full FT can erase general knowledge)
- Confusing **instruction tuning** (behavior change) with **domain adaptation** (knowledge)
- Forgetting that LoRA must be **merged before deployment** for zero inference overhead

---

## Table of Contents
1. [Fine-tuning Methods Overview](#fine-tuning-methods-overview)
2. [Parameter-Efficient Fine-tuning (PEFT)](#parameter-efficient-fine-tuning)
3. [LoRA and Variants](#lora-and-variants)
4. [RLHF Pipeline](#rlhf-pipeline)
5. [Direct Preference Optimization (DPO)](#direct-preference-optimization)
6. [Practical Fine-tuning Guide](#practical-fine-tuning-guide)
7. [Interview Questions](#interview-questions)

---

## 1. Fine-tuning Methods Overview

### Fine-tuning Taxonomy

```python
"""
Fine-tuning Approaches:

1. Full Fine-tuning
   - Update all parameters
   - Best performance, highest cost
   - Risk of catastrophic forgetting

2. Parameter-Efficient Fine-tuning (PEFT)
   - Update small subset of parameters
   - LoRA, Adapters, Prefix-tuning
   - Similar performance, much lower cost

3. Instruction Tuning
   - Fine-tune on instruction-following data
   - FLAN, Alpaca, Vicuna
   - Improves zero-shot capabilities

4. Alignment
   - RLHF: Reinforce from human preferences
   - DPO: Direct optimization from preferences
   - Makes models helpful, harmless, honest
"""

# Method comparison
FINETUNING_METHODS = {
    "full": {
        "trainable_params": "100%",
        "memory": "Very High",
        "performance": "Best",
        "use_case": "Sufficient compute, domain adaptation"
    },
    "lora": {
        "trainable_params": "0.1-1%",
        "memory": "Low",
        "performance": "Near full",
        "use_case": "Limited GPU, multiple tasks"
    },
    "qlora": {
        "trainable_params": "0.1-1%",
        "memory": "Very Low",
        "performance": "Good",
        "use_case": "Consumer GPUs, 7B+ models"
    },
    "prefix_tuning": {
        "trainable_params": "0.1%",
        "memory": "Low",
        "performance": "Good",
        "use_case": "Soft prompts, generation tasks"
    },
    "adapter": {
        "trainable_params": "1-5%",
        "memory": "Medium",
        "performance": "Good",
        "use_case": "Modular, task-specific"
    }
}
```

### Full Fine-tuning

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

dataset = load_dataset("your_dataset")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=True,
    logging_steps=100,
    save_strategy="epoch",
    evaluation_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

trainer.train()
```

---

## 2. Parameter-Efficient Fine-tuning

### Adapter Layers

```python
import torch
import torch.nn as nn

class AdapterLayer(nn.Module):
    """
    Adapter: Bottleneck layer inserted into transformer.
    
    x → LayerNorm → Down-project → Activation → Up-project → (+x)
    
    Only adapter weights are trained.
    
    Paper: "Parameter-Efficient Transfer Learning for NLP" (Houlsby et al., 2019)
    """
    def __init__(self, d_model, bottleneck_dim=64):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return residual + x


class TransformerWithAdapters(nn.Module):
    """Insert adapters after attention and FFN."""
    def __init__(self, base_model, bottleneck_dim=64):
        super().__init__()
        self.base_model = base_model
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Add adapters
        self.adapters = nn.ModuleDict()
        for name, module in self.base_model.named_modules():
            if "attention" in name or "mlp" in name:
                d_model = module.out_features if hasattr(module, 'out_features') else 768
                self.adapters[name] = AdapterLayer(d_model, bottleneck_dim)
```

### Prefix Tuning

```python
class PrefixTuning(nn.Module):
    """
    Prefix Tuning: Prepend learnable "virtual tokens" to input.
    
    Only prefix embeddings are trained.
    
    Input becomes: [PREFIX_1, ..., PREFIX_k, token_1, ..., token_n]
    
    Paper: "Prefix-Tuning" (Li & Liang, 2021)
    """
    def __init__(self, model, prefix_length=20, hidden_dim=512):
        super().__init__()
        self.model = model
        self.prefix_length = prefix_length
        
        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Learnable prefix
        self.d_model = model.config.hidden_size
        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        
        # Prefix embedding (reparameterized through MLP for stability)
        self.prefix_embedding = nn.Embedding(prefix_length, hidden_dim)
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.num_layers * 2 * self.d_model)
        )
    
    def get_prefix(self, batch_size):
        """Generate prefix key-value pairs for each layer."""
        prefix_ids = torch.arange(self.prefix_length).unsqueeze(0).expand(batch_size, -1)
        prefix_emb = self.prefix_embedding(prefix_ids)
        prefix_kv = self.transform(prefix_emb)
        
        # Reshape to (batch, layers, 2, heads, prefix_len, head_dim)
        prefix_kv = prefix_kv.view(
            batch_size, self.prefix_length, self.num_layers, 2, 
            self.num_heads, self.d_model // self.num_heads
        )
        return prefix_kv.permute(2, 3, 0, 4, 1, 5)
    
    def forward(self, input_ids, **kwargs):
        batch_size = input_ids.size(0)
        prefix_kv = self.get_prefix(batch_size)
        
        # Pass prefix to model's attention layers
        return self.model(
            input_ids,
            past_key_values=prefix_kv,
            **kwargs
        )
```

### Prompt Tuning

```python
class PromptTuning(nn.Module):
    """
    Prompt Tuning: Simpler version of prefix tuning.
    
    Only prepend learnable embeddings to input (not to each layer).
    
    Paper: "The Power of Scale for Parameter-Efficient Prompt Tuning" (Lester et al., 2021)
    """
    def __init__(self, model, num_virtual_tokens=20):
        super().__init__()
        self.model = model
        self.num_virtual_tokens = num_virtual_tokens
        
        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Learnable soft prompt
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_virtual_tokens, model.config.hidden_size)
        )
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size = input_ids.size(0)
        
        # Get input embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Prepend soft prompt
        prompt = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([prompt, inputs_embeds], dim=1)
        
        # Adjust attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.num_virtual_tokens, device=attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
```

---

## 3. LoRA and Variants

### LoRA Implementation

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    LoRA: Low-Rank Adaptation.
    
    W' = W + BA where B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)
    
    Key ideas:
    - Freeze original weights W
    - Only train low-rank matrices A, B
    - Scale output by α/r
    
    Paper: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
    """
    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # Original output
        original_output = self.original(x)
        
        # LoRA adaptation: x @ A^T @ B^T * scaling
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        
        return original_output + lora_output * self.scaling
    
    def merge_weights(self):
        """Merge LoRA weights into original for inference."""
        self.original.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        return self.original


def apply_lora_to_model(model, rank=8, alpha=16, target_modules=["q_proj", "v_proj"]):
    """Apply LoRA to specified modules."""
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora_layer)
    
    return model
```

### QLoRA (Quantized LoRA)

```python
"""
QLoRA: LoRA on 4-bit quantized models.

Key innovations:
1. 4-bit NormalFloat (NF4): Better quantization for normally distributed weights
2. Double quantization: Quantize the quantization constants
3. Paged optimizers: Handle memory spikes during gradient checkpointing

Memory savings:
- 7B model: ~28GB → ~6GB
- 70B model: ~280GB → ~48GB
"""

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # NormalFloat4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16
    bnb_4bit_use_double_quant=True          # Quantize quantization constants
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=64,                                    # Higher rank for QLoRA
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Typical output: trainable params: 33M || all params: 3.5B || trainable%: 0.9%
```

### LoRA Variants

```python
"""
LoRA Variants:

1. LoRA (Original)
   - Rank decomposition: W + BA
   - Applied to attention projections

2. LoRA+ 
   - Different learning rates for A and B
   - lr_B = λ * lr_A (λ ~ 2-4)
   - Better convergence

3. DoRA (Weight-Decomposed LoRA)
   - Decompose into magnitude and direction
   - W' = m * (W + BA) / ||W + BA||
   - Better alignment with full fine-tuning

4. QLoRA
   - LoRA on quantized model
   - 4-bit NF4 quantization

5. LoRA-FA (Frozen-A)
   - Freeze A after initialization
   - Only train B
   - Faster, similar performance

6. AdaLoRA
   - Adaptive rank allocation
   - More rank for important layers
   - Pruning during training
"""

class DoRALinear(nn.Module):
    """
    DoRA: Weight-Decomposed Low-Rank Adaptation
    
    Decomposes weight update into magnitude and direction:
    W' = m * (W + BA) / ||W + BA||
    
    where m is learnable magnitude vector
    """
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Freeze original
        self.original.weight.requires_grad = False
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Learnable magnitude
        self.magnitude = nn.Parameter(
            torch.norm(original_layer.weight, dim=1, keepdim=True)
        )
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # Combined weight
        weight = self.original.weight + (self.lora_B @ self.lora_A) * self.scaling
        
        # Normalize and apply magnitude
        weight_norm = torch.norm(weight, dim=1, keepdim=True)
        weight = self.magnitude * weight / weight_norm
        
        return nn.functional.linear(x, weight, self.original.bias)
```

---

## 4. RLHF Pipeline

### Complete RLHF Process

```python
"""
RLHF: Reinforcement Learning from Human Feedback

Three-stage process:
1. Supervised Fine-Tuning (SFT): Train on demonstration data
2. Reward Modeling (RM): Train reward model on comparisons
3. RL Fine-Tuning: Optimize policy with PPO against reward model
"""

# Stage 1: Supervised Fine-Tuning
from transformers import Trainer, TrainingArguments

def sft_training(model, tokenizer, dataset):
    """Train on high-quality instruction-following data."""
    
    def format_example(example):
        return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    
    training_args = TrainingArguments(
        output_dir="./sft_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    return model


# Stage 2: Reward Model Training
import torch
import torch.nn as nn

class RewardModel(nn.Module):
    """
    Reward model: Predicts scalar reward for (prompt, response) pair.
    
    Architecture: LLM backbone + linear head
    Training: Bradley-Terry model on comparison data
    """
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.head = nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Use last token's hidden state
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        reward = self.head(last_hidden)
        return reward


def reward_model_loss(reward_chosen, reward_rejected):
    """
    Bradley-Terry preference model loss.
    
    P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
    Loss = -log(sigmoid(r_chosen - r_rejected))
    """
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()


def train_reward_model(model, comparison_dataset):
    """Train reward model on human preference data."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for batch in comparison_dataset:
        # Get rewards for chosen and rejected responses
        reward_chosen = model(batch['chosen_ids'], batch['chosen_mask'])
        reward_rejected = model(batch['rejected_ids'], batch['rejected_mask'])
        
        loss = reward_model_loss(reward_chosen, reward_rejected)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model


# Stage 3: PPO Training
from trl import PPOTrainer, PPOConfig

def rlhf_training(policy_model, reward_model, tokenizer, prompts):
    """
    PPO training to optimize policy against reward model.
    
    Objective: max E[R(x,y)] - β * KL(π || π_ref)
    
    KL penalty prevents deviation from SFT model.
    """
    ppo_config = PPOConfig(
        model_name="policy_model",
        learning_rate=1.41e-5,
        batch_size=256,
        mini_batch_size=64,
        gradient_accumulation_steps=1,
        ppo_epochs=4,
        kl_penalty="kl",
        init_kl_coef=0.2,              # KL penalty coefficient
        target_kl=6.0,                  # Target KL divergence
        cliprange=0.2,                  # PPO clip range
        cliprange_value=0.2,
    )
    
    # Reference model (frozen SFT model)
    ref_model = create_reference_model(policy_model)
    
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    
    for prompt_batch in prompts:
        # Generate responses
        response_tensors = ppo_trainer.generate(prompt_batch)
        
        # Get rewards
        rewards = [reward_model(r) for r in response_tensors]
        
        # PPO update
        stats = ppo_trainer.step(prompt_batch, response_tensors, rewards)
    
    return policy_model
```

---

## 5. Direct Preference Optimization

### DPO Implementation

```python
import torch
import torch.nn.functional as F

class DPOTrainer:
    """
    DPO: Direct Preference Optimization
    
    Key insight: Optimal policy has closed-form solution given reward
    Can directly optimize policy from preferences without reward model
    
    Loss: -log σ(β(log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
    
    Paper: "Direct Preference Optimization" (Rafailov et al., 2023)
    """
    def __init__(self, model, ref_model, tokenizer, beta=0.1):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def get_log_probs(self, model, input_ids, attention_mask, labels):
        """Get log probabilities of generating response."""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Per-token log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Mask padding
        mask = (shift_labels != self.tokenizer.pad_token_id).float()
        
        # Sum log probs (log probability of sequence)
        return (token_log_probs * mask).sum(-1)
    
    def compute_loss(self, batch):
        """
        DPO loss computation.
        
        batch contains:
        - prompt_ids, prompt_mask
        - chosen_ids, chosen_mask, chosen_labels
        - rejected_ids, rejected_mask, rejected_labels
        """
        # Policy log probs
        pi_chosen_logps = self.get_log_probs(
            self.model, batch['chosen_ids'], batch['chosen_mask'], batch['chosen_labels']
        )
        pi_rejected_logps = self.get_log_probs(
            self.model, batch['rejected_ids'], batch['rejected_mask'], batch['rejected_labels']
        )
        
        # Reference log probs
        with torch.no_grad():
            ref_chosen_logps = self.get_log_probs(
                self.ref_model, batch['chosen_ids'], batch['chosen_mask'], batch['chosen_labels']
            )
            ref_rejected_logps = self.get_log_probs(
                self.ref_model, batch['rejected_ids'], batch['rejected_mask'], batch['rejected_labels']
            )
        
        # DPO loss
        pi_logratios = pi_chosen_logps - pi_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        logits = self.beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(logits).mean()
        
        # Metrics
        chosen_rewards = self.beta * (pi_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (pi_rejected_logps - ref_rejected_logps)
        
        return loss, {
            'chosen_rewards': chosen_rewards.mean().item(),
            'rejected_rewards': rejected_rewards.mean().item(),
            'reward_margin': (chosen_rewards - rejected_rewards).mean().item(),
            'accuracy': (chosen_rewards > rejected_rewards).float().mean().item()
        }


# Using TRL library
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    num_train_epochs=1,
    fp16=True,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### DPO Variants

```python
"""
DPO Variants:

1. DPO (Original)
   - Binary preference: chosen vs rejected
   - Requires reference model

2. IPO (Identity Preference Optimization)
   - More robust to noisy preferences
   - Loss: (log(π/π_ref) - 1/(2β))² instead of log-sigmoid

3. KTO (Kahneman-Tversky Optimization)
   - Only needs positive/negative labels, not pairs
   - Based on prospect theory
   - Asymmetric treatment of gains vs losses

4. ORPO (Odds Ratio Preference Optimization)
   - No reference model needed
   - Combines SFT and preference optimization
   - Uses odds ratio instead of log probability ratio

5. SimPO (Simple Preference Optimization)
   - Length-normalized rewards
   - No reference model
   - Simpler and often better
"""

def ipo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    """IPO: More robust to label noise."""
    log_ratio_chosen = pi_chosen - ref_chosen
    log_ratio_rejected = pi_rejected - ref_rejected
    
    diff = log_ratio_chosen - log_ratio_rejected
    loss = (diff - 1 / (2 * beta)) ** 2
    return loss.mean()


def kto_loss(pi_logps, ref_logps, is_chosen, beta=0.1):
    """KTO: Works with individual samples, not pairs."""
    kl = pi_logps - ref_logps
    
    # Asymmetric treatment
    chosen_mask = is_chosen.float()
    rejected_mask = 1 - chosen_mask
    
    # Different functions for chosen vs rejected
    chosen_loss = 1 - F.sigmoid(beta * kl)
    rejected_loss = 1 - F.sigmoid(-beta * kl)
    
    loss = chosen_mask * chosen_loss + rejected_mask * rejected_loss
    return loss.mean()


def orpo_loss(pi_chosen, pi_rejected, beta=0.1):
    """ORPO: No reference model needed."""
    # Odds ratio
    odds_chosen = pi_chosen / (1 - pi_chosen + 1e-10)
    odds_rejected = pi_rejected / (1 - pi_rejected + 1e-10)
    
    log_odds_ratio = torch.log(odds_chosen / odds_rejected)
    
    # Combined SFT + preference loss
    sft_loss = -pi_chosen.mean()  # Standard NLL
    pref_loss = -F.logsigmoid(beta * log_odds_ratio).mean()
    
    return sft_loss + pref_loss
```

---

## 6. Practical Fine-tuning Guide

### Complete Fine-tuning Script

```python
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import wandb

# Configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
OUTPUT_DIR = "./fine_tuned_model"
USE_QLORA = True

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model
if USE_QLORA:
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

# LoRA configuration
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Dataset preparation
def format_instruction(example):
    """Format for instruction tuning."""
    if example.get("input"):
        text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        text = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
    
    return {"text": text}

dataset = load_dataset("tatsu-lab/alpaca")
dataset = dataset.map(format_instruction)

def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=2048,
        padding=False
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset["train"].map(
    tokenize,
    remove_columns=dataset["train"].column_names
)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    max_grad_norm=0.3,
    report_to="wandb"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Train
trainer.train()

# Save
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

---

## 7. Interview Questions

```python
# Q1: What is LoRA and why is it effective?
"""
LoRA: Low-Rank Adaptation

Key idea: Weight update ΔW has low intrinsic rank
W' = W + ΔW = W + BA where B ∈ R^{d×r}, A ∈ R^{r×k}

Why effective:
1. Dramatic parameter reduction: 0.1% of original
2. Weight changes are low-rank in practice
3. Can merge back for no inference overhead
4. Task-specific adapters can be swapped

Training: Only A, B updated (r << d, k)
Inference: Merge W' = W + BA, no extra cost
"""


# Q2: Compare RLHF and DPO
"""
RLHF:
- 3 stages: SFT → Reward Model → PPO
- Requires reward model training
- Online RL (generates during training)
- More compute intensive
- Can overfit reward model

DPO:
- 2 stages: SFT → Direct preference optimization
- No reward model needed
- Offline (uses fixed preference dataset)
- Simpler, more stable
- Closed-form solution

DPO shows: Reward modeling and RL can be combined
into single supervised loss, achieving similar results
"""


# Q3: What is the purpose of the KL penalty in RLHF?
"""
KL penalty: β * KL(π || π_ref)

Purpose:
1. Prevent reward hacking (gaming reward model)
2. Maintain linguistic quality from SFT
3. Regularization against distribution shift
4. Keep outputs diverse (avoid mode collapse)

Without KL penalty:
- Model may find adversarial outputs that fool reward model
- Quality degrades while reward increases
- Outputs become repetitive or degenerate
"""


# Q4: Explain QLoRA's memory savings
"""
QLoRA memory breakdown:

Standard 7B model:
- 7B × 4 bytes (FP32) = 28GB
- Or 7B × 2 bytes (FP16) = 14GB

QLoRA:
- Base model: 7B × 0.5 bytes (4-bit) = 3.5GB
- LoRA adapters: ~100M × 2 bytes = 0.2GB
- Optimizer states (LoRA only): ~0.4GB
- Activations: ~2GB
- Total: ~6GB

Key techniques:
1. NF4 quantization (information-theoretic optimal)
2. Double quantization (quantize scale factors)
3. Paged optimizers (handle memory spikes)
"""


# Q5: When to use full fine-tuning vs LoRA?
"""
Use Full Fine-tuning when:
- Sufficient compute resources
- Target domain very different from pretraining
- Maximum performance required
- Single-task deployment

Use LoRA when:
- Limited GPU memory
- Multiple task adapters needed
- Quick iteration required
- Want to preserve base model

Use QLoRA when:
- Consumer GPU (24GB or less)
- Large models (7B+)
- Memory is primary constraint

In practice: LoRA achieves 90-95% of full fine-tuning
performance with <1% parameters
"""


# Q6: What is catastrophic forgetting and how to prevent it?
"""
Catastrophic forgetting: Model loses pretrained capabilities
after fine-tuning on new task

Prevention strategies:
1. Lower learning rate (1e-5 to 5e-5)
2. Short training (few epochs)
3. LoRA (original weights unchanged)
4. Regularization toward pretrained weights
5. Replay buffer (mix in pretraining data)
6. Elastic Weight Consolidation (EWC)

LoRA naturally prevents forgetting:
- Original weights W are frozen
- Only small ΔW = BA is learned
- Can remove adaptation to restore original
"""


# Q7: Explain the reward model training process
"""
Reward model training:

Data format: (prompt, chosen_response, rejected_response)
From human annotators comparing outputs

Architecture:
- LLM backbone (often same as policy)
- Scalar head on last token
- Output: single reward value

Loss (Bradley-Terry):
L = -log σ(r_chosen - r_rejected)

Intuition: Probability that chosen is better than rejected
should be high → difference in rewards should be positive

Tips:
- Use same tokenizer as policy
- Train on diverse prompts
- Balance chosen/rejected quality
"""


# Q8: What are the challenges of RLHF?
"""
Challenges:

1. Reward hacking
- Model exploits reward model flaws
- Solution: KL penalty, reward model ensembles

2. Training instability
- PPO can be unstable
- Solution: Careful hyperparameters, gradient clipping

3. Expensive data collection
- Need human comparisons
- Solution: AI feedback (RLAIF), synthetic data

4. Reward model limitations
- May not capture all preferences
- Solution: Multi-objective rewards, Constitutional AI

5. Computational cost
- Multiple models (policy, reference, reward)
- Solution: DPO (no reward model), efficient implementations
"""


# Q9: How does instruction tuning improve zero-shot performance?
"""
Instruction tuning: Fine-tune on diverse instruction-following data

Examples (FLAN, Alpaca):
- "Summarize this text: ..."
- "Translate to French: ..."
- "Answer this question: ..."

Why it works:
1. Teaches model to follow instructions
2. Generalizes to unseen instructions
3. Activates pretrained knowledge
4. Aligns output format expectations

Key insights:
- Diversity of tasks matters more than quantity
- Natural instructions > templates
- Chain-of-thought examples improve reasoning
"""


# Q10: Compare different PEFT methods
"""
Method comparison:

Adapters:
- Insert bottleneck layers
- 1-5% parameters
- Inference overhead
- Good for classification

LoRA:
- Low-rank weight updates
- 0.1-1% parameters
- No inference overhead (merge)
- Good for generation

Prefix Tuning:
- Learnable soft prompts at each layer
- 0.1% parameters
- Some inference overhead
- Good for generation

Prompt Tuning:
- Soft prompts at input only
- <0.1% parameters
- Minimal overhead
- Works best at scale (>10B)

Recommendation:
- Default choice: LoRA (best balance)
- Very limited memory: QLoRA
- Multiple tasks: Adapters (modular)
"""
```

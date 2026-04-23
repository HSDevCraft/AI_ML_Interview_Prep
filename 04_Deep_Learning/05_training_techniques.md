# Deep Learning Training Techniques - Complete Guide

## Table of Contents
1. [Training Loop Best Practices](#training-loop-best-practices)
2. [Data Loading and Augmentation](#data-loading-and-augmentation)
3. [Transfer Learning](#transfer-learning)
4. [Distributed Training](#distributed-training)
5. [Model Compression](#model-compression)
6. [Debugging and Monitoring](#debugging-and-monitoring)
7. [Interview Questions](#interview-questions)

---

## 1. Training Loop Best Practices

### Complete Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import wandb

class Trainer:
    """Production-ready training loop."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device='cuda',
        config=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.get('use_amp', False) else None
        
        # Gradient accumulation
        self.accumulation_steps = config.get('accumulation_steps', 1)
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = config.get('patience', 5)
        
        # Logging
        self.use_wandb = config.get('use_wandb', False)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                output = self.model(data)
                loss = self.criterion(output, target)
                loss = loss / self.accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.config.get('grad_clip', 0) > 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip']
                    )
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # LR scheduler step (per batch for OneCycleLR)
                if self.scheduler and isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()
            
            total_loss += loss.item() * self.accumulation_steps
            pbar.set_postfix({'loss': loss.item() * self.accumulation_steps})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in self.val_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            output = self.model(data)
            loss = self.criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        return total_loss / len(self.val_loader), correct / total
    
    def fit(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            # LR scheduler step (per epoch)
            if self.scheduler and not isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step(val_loss)
            
            # Logging
            print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, '
                  f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')
            
            if self.use_wandb:
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        return self.model
```

### Learning Rate Scheduling

```python
import torch.optim.lr_scheduler as lr_scheduler

# Step LR: Decay by gamma every step_size epochs
step_lr = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Multi-step LR: Decay at specific epochs
multistep_lr = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

# Exponential decay
exp_lr = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Cosine annealing
cosine_lr = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Cosine annealing with warm restarts
cosine_warm = lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

# One cycle policy (best for training from scratch)
one_cycle = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    total_steps=len(train_loader) * num_epochs,
    pct_start=0.3,  # Warmup fraction
    anneal_strategy='cos'
)

# Reduce on plateau (for validation-based scheduling)
plateau_lr = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10
)

# Linear warmup + decay (common for transformers)
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - num_warmup_steps))
        )
    return lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Gradient Clipping

```python
# Clip by norm (most common)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# Custom gradient clipping
def clip_gradients(model, max_norm=1.0):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    
    return total_norm
```

---

## 2. Data Loading and Augmentation

### Efficient Data Loading

```python
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms

class CustomDataset(Dataset):
    """Standard map-style dataset."""
    
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


# Optimized DataLoader settings
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,           # Parallel data loading
    pin_memory=True,         # Faster GPU transfer
    prefetch_factor=2,       # Prefetch batches
    persistent_workers=True, # Keep workers alive between epochs
    drop_last=True           # Consistent batch size for BatchNorm
)


# Iterable dataset for large/streaming data
class StreamingDataset(IterableDataset):
    """For data that doesn't fit in memory or is streaming."""
    
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            files = self.file_paths
        else:
            # Split files among workers
            per_worker = len(self.file_paths) // worker_info.num_workers
            worker_id = worker_info.id
            files = self.file_paths[worker_id * per_worker:(worker_id + 1) * per_worker]
        
        for file_path in files:
            for sample in self.load_file(file_path):
                if self.transform:
                    sample = self.transform(sample)
                yield sample
```

### Image Augmentation

```python
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Torchvision transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5)
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Albumentations (faster, more augmentations)
train_transform_alb = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(p=0.2),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


# MixUp augmentation
def mixup(x, y, alpha=0.2):
    """MixUp: Convex combination of samples."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# CutMix augmentation
def cutmix(x, y, alpha=1.0):
    """CutMix: Cut and paste patches."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    # Random box
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda based on actual box size
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y, y[index], lam
```

---

## 3. Transfer Learning

### Fine-tuning Pre-trained Models

```python
import torch
import torch.nn as nn
from torchvision import models

# Load pre-trained model
model = models.resnet50(pretrained=True)

# Strategy 1: Replace final layer only
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Freeze all layers except final
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True


# Strategy 2: Gradual unfreezing
def unfreeze_layers(model, num_layers):
    """Unfreeze last num_layers."""
    layers = list(model.children())
    for layer in layers[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True


# Strategy 3: Discriminative learning rates
def get_parameter_groups(model, lr_base=1e-4, lr_mult=0.1):
    """Lower LR for earlier layers, higher for later."""
    params = []
    
    # Backbone with lower LR
    params.append({
        'params': model.conv1.parameters(),
        'lr': lr_base * lr_mult ** 3
    })
    params.append({
        'params': model.layer1.parameters(),
        'lr': lr_base * lr_mult ** 3
    })
    params.append({
        'params': model.layer2.parameters(),
        'lr': lr_base * lr_mult ** 2
    })
    params.append({
        'params': model.layer3.parameters(),
        'lr': lr_base * lr_mult
    })
    params.append({
        'params': model.layer4.parameters(),
        'lr': lr_base
    })
    
    # Head with full LR
    params.append({
        'params': model.fc.parameters(),
        'lr': lr_base
    })
    
    return params

optimizer = torch.optim.AdamW(get_parameter_groups(model))


# Strategy 4: Feature extraction + classifier
class FeatureExtractor(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        
        # Freeze feature extractor
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Trainable classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)
```

### Transfer Learning for NLP

```python
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained transformer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2
)

# Freeze BERT, train classifier only
for param in model.bert.parameters():
    param.requires_grad = False

# Or freeze specific layers
for layer in model.bert.encoder.layer[:8]:  # Freeze first 8 layers
    for param in layer.parameters():
        param.requires_grad = False


# LoRA (Low-Rank Adaptation) for efficient fine-tuning
"""
Instead of fine-tuning all parameters, add low-rank matrices:
W = W_0 + BA where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)

Benefits:
- Much fewer trainable parameters
- Can switch adapters for different tasks
- Original model unchanged

pip install peft
"""
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                       # Rank
    lora_alpha=32,             # Scaling
    target_modules=['q_proj', 'v_proj'],  # Modules to adapt
    lora_dropout=0.1
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Shows % of trainable params
```

---

## 4. Distributed Training

### Data Parallel

```python
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

# DataParallel (single machine, multiple GPUs)
# Simple but limited scalability
model = nn.DataParallel(model)
model = model.to('cuda')

# Forward pass automatically splits batch across GPUs
output = model(input)  # Batch split, processed in parallel, gathered


# DistributedDataParallel (recommended)
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def setup_ddp(rank, world_size):
    """Initialize distributed training."""
    dist.init_process_group(
        backend='nccl',  # 'nccl' for GPU, 'gloo' for CPU
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size, model, dataset, epochs):
    setup_ddp(rank, world_size)
    
    model = model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    
    # Distributed sampler ensures no overlap between processes
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch
        
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
    
    cleanup()

# Launch with:
# torchrun --nproc_per_node=4 train.py
# or
# python -m torch.distributed.launch --nproc_per_node=4 train.py
```

### Mixed Precision Training

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Automatic Mixed Precision (AMP)
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward pass in mixed precision
    with autocast():
        output = model(batch)
        loss = criterion(output, target)
    
    # Backward pass with scaling
    scaler.scale(loss).backward()
    
    # Unscale gradients for clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step with scaler
    scaler.step(optimizer)
    scaler.update()


# Benefits of mixed precision:
"""
1. Memory: FP16 uses half the memory of FP32
2. Speed: Tensor cores accelerate FP16 operations
3. Accuracy: Loss scaling prevents underflow

Typical speedup: 1.5-3x depending on model
"""
```

### Gradient Accumulation

```python
# Simulate larger batch size without more memory
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

optimizer.zero_grad()

for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target)
    
    # Normalize loss by accumulation steps
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 5. Model Compression

### Pruning

```python
import torch.nn.utils.prune as prune

# Magnitude pruning (remove smallest weights)
model = MyModel()

# Prune 30% of weights in specific layer
prune.l1_unstructured(model.fc1, name='weight', amount=0.3)

# Prune globally (across all layers)
parameters_to_prune = [
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
]
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.4
)

# Remove pruning reparametrization (make permanent)
prune.remove(model.fc1, 'weight')

# Check sparsity
def get_sparsity(model):
    zeros = 0
    total = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            zeros += (param == 0).sum().item()
            total += param.numel()
    return zeros / total


# Structured pruning (remove entire channels/neurons)
prune.ln_structured(model.conv1, name='weight', amount=0.3, n=2, dim=0)
```

### Quantization

```python
import torch.quantization as quant

# Post-training quantization
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Dynamic quantization (weights quantized, activations at runtime)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.LSTM},
    dtype=torch.qint8
)

# Static quantization (both weights and activations pre-quantized)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model)

# Calibrate with representative data
for batch in calibration_loader:
    model_prepared(batch)

model_quantized = torch.quantization.convert(model_prepared)


# Quantization-Aware Training (QAT)
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_prepared = torch.quantization.prepare_qat(model)

# Train with fake quantization
for epoch in range(num_epochs):
    train(model_prepared)

model_quantized = torch.quantization.convert(model_prepared)
```

### Knowledge Distillation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss.
    
    L = α * L_hard + (1-α) * T² * L_soft
    
    L_hard: Cross-entropy with true labels
    L_soft: KL divergence with teacher's soft targets
    T: Temperature (higher = softer probabilities)
    """
    
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, targets):
        # Hard loss (with true labels)
        hard_loss = self.ce_loss(student_logits, targets)
        
        # Soft loss (with teacher's predictions)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(soft_student, soft_teacher)
        
        # Combined loss
        return (
            self.alpha * hard_loss + 
            (1 - self.alpha) * (self.temperature ** 2) * soft_loss
        )


# Training with distillation
teacher = TeacherModel()
teacher.load_state_dict(torch.load('teacher.pth'))
teacher.eval()

student = StudentModel()
distill_loss = DistillationLoss(temperature=4.0, alpha=0.5)
optimizer = torch.optim.Adam(student.parameters())

for batch, targets in dataloader:
    with torch.no_grad():
        teacher_logits = teacher(batch)
    
    student_logits = student(batch)
    loss = distill_loss(student_logits, teacher_logits, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 6. Debugging and Monitoring

### Debugging Tools

```python
# Gradient checking
def check_gradients(model, loss):
    """Check for vanishing/exploding gradients."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f'{name}: grad_norm = {grad_norm:.6f}')
            
            if grad_norm < 1e-7:
                print(f'  WARNING: Vanishing gradient!')
            elif grad_norm > 1e3:
                print(f'  WARNING: Exploding gradient!')


# Activation statistics
class ActivationStats:
    """Track activation statistics through forward hooks."""
    
    def __init__(self, model):
        self.stats = {}
        self.hooks = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(
                    self._get_activation_hook(name)
                )
                self.hooks.append(hook)
    
    def _get_activation_hook(self, name):
        def hook(module, input, output):
            self.stats[name] = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'max': output.max().item(),
                'min': output.min().item(),
                'zeros': (output == 0).float().mean().item()
            }
        return hook
    
    def print_stats(self):
        for name, stat in self.stats.items():
            print(f'{name}:')
            print(f'  mean={stat["mean"]:.4f}, std={stat["std"]:.4f}')
            print(f'  range=[{stat["min"]:.4f}, {stat["max"]:.4f}]')
            print(f'  zeros={stat["zeros"]*100:.1f}%')


# Detect anomalies
torch.autograd.set_detect_anomaly(True)  # Catch NaN/Inf in backward

# Check for NaN/Inf
def check_nan_inf(tensor, name='tensor'):
    if torch.isnan(tensor).any():
        print(f'NaN detected in {name}')
    if torch.isinf(tensor).any():
        print(f'Inf detected in {name}')
```

### Logging and Visualization

```python
# TensorBoard logging
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

for epoch in range(num_epochs):
    # Log scalars
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
    
    # Log histograms
    for name, param in model.named_parameters():
        writer.add_histogram(f'params/{name}', param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'grads/{name}', param.grad, epoch)
    
    # Log images
    writer.add_images('samples', images[:8], epoch)
    
    # Log model graph
    if epoch == 0:
        writer.add_graph(model, sample_input)

writer.close()


# Weights & Biases
import wandb

wandb.init(
    project='my-project',
    config={
        'learning_rate': 0.001,
        'batch_size': 32,
        'architecture': 'ResNet50'
    }
)

for epoch in range(num_epochs):
    wandb.log({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'epoch': epoch
    })
    
    # Log images
    wandb.log({
        'examples': [wandb.Image(img) for img in images[:5]]
    })

wandb.finish()
```

### Profiling

```python
# PyTorch Profiler
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("model_inference"):
        model(input)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for visualization
prof.export_chrome_trace("trace.json")


# Memory profiling
print(torch.cuda.memory_summary())

# Track memory allocation
torch.cuda.reset_peak_memory_stats()
# ... run model ...
peak_memory = torch.cuda.max_memory_allocated() / 1e9
print(f'Peak memory: {peak_memory:.2f} GB')
```

---

## 7. Interview Questions

```python
# Q1: What is the difference between DataParallel and DistributedDataParallel?
"""
DataParallel:
- Single process, multiple threads
- Model replicated on each GPU
- Gradients gathered on GPU 0 (bottleneck)
- Simpler but slower

DistributedDataParallel:
- Multiple processes (one per GPU)
- All-reduce for gradient sync
- No single GPU bottleneck
- Faster, more scalable
- Recommended for multi-GPU training
"""


# Q2: Explain mixed precision training
"""
Mixed precision uses FP16 for most operations, FP32 for critical ones:

1. Weights stored in FP32 (master copy)
2. Forward pass in FP16
3. Loss computed in FP32
4. Backward pass in FP16
5. Gradients converted to FP32, applied to master weights

Benefits:
- 2x memory reduction
- 2-4x speed improvement (Tensor Cores)

Challenges:
- Underflow (small gradients → 0)
- Solution: Loss scaling
"""


# Q3: When would you use gradient accumulation?
"""
Use gradient accumulation when:
1. Batch size is limited by GPU memory
2. Model requires large effective batch size for stability
3. Distributed training isn't available

Example: Want batch_size=256 but only fit 64
→ Accumulate gradients over 4 steps

Caveat: BatchNorm behaves differently (normalizes per micro-batch)
"""


# Q4: What is knowledge distillation and when to use it?
"""
Knowledge distillation: Train small student to mimic large teacher

Process:
1. Train large teacher model
2. Student learns from teacher's soft outputs (not just hard labels)
3. Temperature softens probability distribution
4. Soft targets contain more information (dark knowledge)

Use when:
- Need smaller model for deployment
- Want to compress ensemble into single model
- Student architecture differs from teacher
"""


# Q5: How do you choose a learning rate schedule?
"""
Common schedules:

1. Step decay: Simple, for convnets
2. Cosine annealing: Smooth, modern default
3. One Cycle: Best for training from scratch
4. Linear warmup + decay: Standard for transformers
5. Reduce on plateau: Adaptive, for fine-tuning

Tips:
- Always use warmup for transformers
- One Cycle often fastest convergence
- Monitor validation loss for adaptive schedules
"""


# Q6: What are common causes of training instability?
"""
1. Learning rate too high → exploding gradients
2. Bad initialization → vanishing/exploding gradients
3. No gradient clipping for RNNs/Transformers
4. Missing BatchNorm/LayerNorm
5. Data issues (wrong normalization, label noise)

Solutions:
- Gradient clipping
- Lower learning rate + warmup
- Proper initialization (He, Xavier)
- Add normalization layers
- Check data pipeline
"""


# Q7: How do you diagnose overfitting vs underfitting?
"""
Overfitting:
- Train loss decreasing, val loss increasing
- Large gap between train and val performance

Solutions:
- More data / augmentation
- Regularization (dropout, weight decay)
- Early stopping
- Simpler model

Underfitting:
- Both train and val loss high
- Model not learning

Solutions:
- Larger / more complex model
- Train longer
- Higher learning rate
- Less regularization
"""


# Q8: What is the purpose of learning rate warmup?
"""
Warmup: Start with small LR, gradually increase to target

Why needed:
1. Early gradients are noisy (random initialization)
2. Large updates can destabilize training
3. Adam moments need time to stabilize
4. Especially important for transformers (large models)

Typical warmup: 1-10% of total steps
"""


# Q9: How does pruning reduce model size?
"""
Pruning removes unnecessary weights:

Unstructured pruning:
- Remove individual weights (sparse matrix)
- High compression, needs special hardware

Structured pruning:
- Remove entire channels/neurons
- Lower compression, works on any hardware

Methods:
1. Magnitude pruning (remove small weights)
2. Gradient-based (remove low gradient weights)
3. Learned (learn mask during training)
"""


# Q10: What metrics should you monitor during training?
"""
Essential:
- Train/val loss
- Train/val accuracy (or task metric)
- Learning rate
- Gradient norm

Advanced:
- Activation distributions
- Weight distributions
- GPU memory usage
- Training speed (samples/sec)

Red flags:
- Loss going to NaN/Inf
- Gradient norm exploding/vanishing
- Activations saturating (all 0 or 1)
"""
```

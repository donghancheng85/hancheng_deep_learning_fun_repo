Our custom ViT model architecture closely mimics that of the ViT paper, however, our training recipe misses a few things. Research some of the following topics from Table 3 in the ViT paper that we miss and write a sentence about each and how it might help with training:

## ImageNet-21k pretraining (more data)

The ViT paper shows that ViT only outperforms CNNs when trained on very large datasets (14M–300M images); on small datasets CNNs still win because ViT lacks the inductive biases (locality, translation equivariance) that CNNs have built in. Pretraining on ImageNet-21k (14M images, 21,843 classes) before fine-tuning gives the model a much richer and more general feature space, so it generalises well even when the downstream dataset is tiny — exactly the situation we're in with FoodVision Mini (~675 training images).

**In our code:** We rely on `torchvision` pretrained weights which are already trained on ImageNet-1k (1.28M images). Switching to a model pretrained on ImageNet-21k (e.g., via `timm`) would give a stronger starting point and likely push accuracy even higher.

## Learning rate warmup

Learning rate warmup linearly increases the learning rate from near-zero to the target value over the first few thousand steps, before the main schedule takes over. Without warmup, the randomly initialised classification head produces large, noisy gradients at the start of training that can destabilise the pretrained backbone weights (if unfrozen) or cause the optimiser's momentum/variance estimates to be corrupted early on. Warmup gives the optimiser time to build reliable gradient statistics before the full learning rate kicks in.

**In our code:** We jump straight to `lr=1e-3` at step 0. We could add warmup with PyTorch's `LinearLR` scheduler chained before `CosineAnnealingLR`:

```python
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
)
main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, main], milestones=[warmup_steps]
)
```

## Learning rate decay

Learning rate decay (most commonly cosine annealing in the ViT paper) gradually reduces the learning rate towards zero as training progresses. A high learning rate early on helps the model escape poor initialisations quickly; a low learning rate late in training allows fine-grained weight adjustments without overshooting minima. The ViT paper uses cosine decay for the full training run.

**In our code:** We already use `CosineAnnealingLR`, so this one is covered. The main gap is that our `T_max=EPOCHS` (10) means the cosine cycle completes after just 10 steps — in the ViT paper they train for thousands of steps. With more epochs, cosine decay becomes more meaningful.

## Gradient clipping

Gradient clipping caps the norm of the gradient vector to a maximum value (typically 1.0) before the optimiser step. Transformers are particularly prone to gradient explosions because the attention mechanism can amplify gradients across many layers. Clipping prevents a single bad batch from causing a catastrophic weight update, stabilising training especially early on and when using larger learning rates.

**In our code:** We have no gradient clipping. It can be added with a single line before `optimizer.step()` in the training loop:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

In `engine.py` the train step would become:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```
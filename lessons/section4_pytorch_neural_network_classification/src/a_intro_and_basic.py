"""
Example code before start learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Reproducibility
torch.manual_seed(42)

# -------------------------------------------------
# 1. Create synthetic dataset
# -------------------------------------------------
# Two classes:
# Class 0: centered near (-2, -2)
# Class 1: centered near (2, 2)

n_samples = 200

X_class0 = torch.randn(n_samples, 2) + torch.tensor([-2.0, -2.0])
X_class1 = torch.randn(n_samples, 2) + torch.tensor([2.0, 2.0])

X = torch.cat([X_class0, X_class1], dim=0)

y_class0 = torch.zeros(n_samples)
y_class1 = torch.ones(n_samples)
y = torch.cat([y_class0, y_class1], dim=0)

# Important: BCEWithLogitsLoss expects float targets
y = y.unsqueeze(1)  # shape: (400, 1)


# -------------------------------------------------
# 2. Define simple neural network
# -------------------------------------------------
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),  # 2 input features → 16 hidden units
            nn.ReLU(),
            nn.Linear(16, 1),  # Output 1 logit (NOT sigmoid!)
        )

    def forward(self, x):
        return self.model(x)  # raw logits


model = BinaryClassifier()

# -------------------------------------------------
# 3. Define loss + optimizer
# -------------------------------------------------
# IMPORTANT:
# We use BCEWithLogitsLoss instead of Sigmoid + BCELoss
loss_fn = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

# -------------------------------------------------
# 4. Training loop
# -------------------------------------------------
epochs = 200

for epoch in range(epochs):

    model.train()

    # Forward pass
    logits = model(X)  # shape: (400, 1)

    loss = loss_fn(logits, y)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print occasionally
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# -------------------------------------------------
# 5. Evaluation
# -------------------------------------------------
model.eval()
with torch.inference_mode():
    logits = model(X)
    probs = torch.sigmoid(logits)  # convert logits → probabilities
    preds = (probs > 0.5).float()  # threshold at 0.5

    accuracy = (preds == y).float().mean()

print(f"\nFinal Accuracy: {accuracy.item() * 100:.2f}%")

# -------------------------------------------------
# 6. Visualization
# -------------------------------------------------
model.eval()

# Create mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = torch.meshgrid(
    torch.linspace(x_min, x_max, 200), torch.linspace(y_min, y_max, 200), indexing="xy"
)

grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

with torch.inference_mode():
    logits = model(grid)
    probs = torch.sigmoid(logits)
    Z = probs.reshape(xx.shape)

# Convert to numpy for plotting
xx_np = xx.numpy()
yy_np = yy.numpy()
Z_np = Z.numpy()

X_np = X.numpy()
y_np = y.numpy().flatten()

# Plot decision surface
plt.figure(figsize=(8, 6))

# Background probability contour
plt.contourf(xx_np, yy_np, Z_np, levels=50, cmap="coolwarm", alpha=0.6)

# Decision boundary at probability=0.5
plt.contour(xx_np, yy_np, Z_np, levels=[0.5], colors="black")

# Scatter data points
plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap="coolwarm", edgecolors="k")

plt.title("Binary Classification Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Predicted Probability (Class 1)")
plt.savefig(
    "lessons/section4_pytorch_neural_network_classification/src/a_demo_figure.png"
)

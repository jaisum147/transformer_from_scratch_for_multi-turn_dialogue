import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import DialogueDataset
from model import TransformerClassifier

# Load dataset
dataset = DialogueDataset("train.csv")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model
model = TransformerClassifier(
    vocab_size=len(dataset.vocab),
    d_model=64,
    num_heads=4,
    num_layers=2,
    num_classes=7
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(3):
    total_loss = 0
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Test forward pass
sample_x, _ = next(iter(loader))
output = model(sample_x)
print("Output shape:", output.shape)

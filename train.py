import torch
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn as nn

from dataset import TransformerDataset
from transformer_timeseries import TimeSeriesTransformer
from pytorch_lightning import Trainer
from pytorch_forecasting.metrics import SMAPE
from torch.utils.data import DataLoader

# load data
data = np.load("final/lorenz63_on0.05_train.npy")
# data = np.load("final/lorenz63_test.npy")
# Convert to PyTorch tensor
data_tensor = torch.tensor(data)
print("Data shape:", data_tensor.shape)

# Define sequence lengths
enc_seq_len = 50  # History window
dec_seq_len = 10  # Decoder input length
target_seq_len = 10  # Future steps to predict

# Generate indices for slicing the data
indices = [(i, i + enc_seq_len + target_seq_len) for i in range(10000 - (enc_seq_len + target_seq_len))]
print(f"Generated {len(indices)} sequences.")

# Create dataset
dataset = TransformerDataset(
    data=data_tensor,
    indices=indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=target_seq_len
)

# Wrap in DataLoader for batching
batch_size = 64  # Adjust based on your hardware
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Test a single batch
src, trg, trg_y = next(iter(dataloader))

# print(f"Encoder Input (src) Shape: {src.shape}")  # Expected (batch_size, 50, 3)
# print(f"Decoder Input (trg) Shape: {trg.shape}")  # Expected (batch_size, 10, 3)
# print(f"Target Output (trg_y) Shape: {trg_y.shape}")  # Expected (batch_size, 10, 3)





model = TimeSeriesTransformer(
    input_size=3,
    dec_seq_len=dec_seq_len,
    batch_first=True,
)

# Check if MPS is available, otherwise fall back to CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Move model to MPS
model.to(device)
print(f"Model initialized on: {device}")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#training loop
num_epochs = 10  # Adjust as needed
save_path = "fine_tuned_transformer.pth"

print("\nStarting training...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for src, trg, trg_y in dataloader:
        # Move data to device
        src, trg, trg_y = src.to(device), trg.to(device), trg_y.to(device)

        optimizer.zero_grad()  # Reset gradients

        # Create masks (if required by the model)
        src_mask = None
        tgt_mask = None

        # Forward pass
        output = model(src, trg, src_mask, tgt_mask)

        # Ensure output shape matches trg_y to avoid shape mismatch in Linear layers
        if output.shape != trg_y.shape:
            print(f"Warning: output shape {output.shape} does not match trg_y {trg_y.shape}, adjusting shape...")
            output = output[:, : trg_y.shape[1], :]  # Trim excess dimensions

        # Compute loss
        loss = criterion(output, trg_y)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

# 7. Save the Trained Model
torch.save(model.state_dict(), save_path)
print(f"\nTraining complete, model saved as {save_path}")

# 8. Quick Validation
print("\nRunning quick validation...")
model.eval()

with torch.no_grad():
    src, trg, trg_y = dataset[0]  # Get the first sample
    src, trg = src.unsqueeze(0).to(device), trg.unsqueeze(0).to(device)  # Add batch dimension

    prediction = model(src, trg)  # Generate prediction

print(f"Validation complete, output shape: {prediction.shape}")  # Should match (1, 10, 3)
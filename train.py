import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TransformerDataset
from transformer_timeseries import TimeSeriesTransformer
import utils
from psd import combined_loss


# 1. Load Data

data_path = "finalProject/DSML/lorenz63_on0.05_train.npy"
data = np.load(data_path)

# Convert to PyTorch tensor
data_tensor = torch.tensor(data, dtype=torch.float32)
print("Data shape:", data_tensor.shape)  # Expected (100000, 3)


# 2. Define Sequence Lengths & Indexing

enc_seq_len = 40  # Encoder input length
dec_seq_len = 20  # Decoder input length
target_seq_len = 20  # Prediction length
step_size = 1  
forecast_horizon = 1  

# Generate indices for slicing the data
indices = utils.get_indices_input_target(
    num_obs=len(data), 
    input_len=enc_seq_len + dec_seq_len,
    step_size=step_size, 
    forecast_horizon=forecast_horizon, 
    target_len=target_seq_len
)
print(f"Generated {len(indices)} training sequences.")
print(indices[:5])

# 3. Create Dataset & DataLoader

dataset = TransformerDataset(
    data=data_tensor,
    indices=indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=target_seq_len
)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

src, trg, trg_y = next(iter(dataloader))

# 4. Initialize Model

model = TimeSeriesTransformer(
    input_size=3,
    dec_seq_len=dec_seq_len,
    batch_first=True,
)

# Select device: MPS (Mac) or CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)
print(f"Model initialized on: {device}")


# 5. Define Loss & Optimizer

criterion = combined_loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Gradient Clipping
clip_value = 1.0


# 6. Training Loop

num_epochs = 10  
save_path = "fine_tuned_transformer.pth"

print("\nStarting training...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for  src, trg, trg_y in dataloader:
        # Move data to device
        src, trg, trg_y = src.to(device), trg.to(device), trg_y.to(device)

        optimizer.zero_grad()  # Reset gradients

        # Create masks
        src_mask = utils.generate_square_subsequent_mask(target_seq_len, enc_seq_len).to(device)
        trg_mask = utils.generate_square_subsequent_mask(target_seq_len, target_seq_len).to(device)

        # Forward pass
        output = model(src, trg, src_mask, trg_mask)

        # Ensure output shape matches target shape
        if output.shape != trg_y.shape:
            print(f"Warning: output shape {output.shape} does not match trg_y {trg_y.shape}, adjusting shape...")
            output = output[:, : trg_y.shape[1], :]

        # Compute loss
        loss = criterion(output.to(device), trg_y.to(device), alpha=1.0, beta=0.5, gamma=0.1)
        loss.backward()  # Backpropagation

        # Gradient Clipping (to prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()  # Update weights
        model.eval()
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

print(f"Validation complete, output shape: {prediction.shape}")  # Expected (1, 10, 3)

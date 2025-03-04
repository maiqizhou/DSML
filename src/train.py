import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import TransformerDataset
from transformer_timeseries import TimeSeriesTransformer
import utils


# 1. Load Data
data_path = "finalProject/DSML/lorenz96_on0.05_train.npy"
data = np.load(data_path)

# Convert to PyTorch tensor
data_tensor = torch.tensor(data, dtype=torch.float32)

# 2. Define Sequence Lengths & Indexing
enc_seq_len = 20  # Encoder input length
dec_seq_len = 10  # Decoder input length
target_seq_len = 10  # Prediction length
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

# **Split indices into 80% training and 20% validation**
split_idx = int(0.8 * len(indices))
train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

# **Create Train & Validation Datasets**
train_dataset = TransformerDataset(
    data=data_tensor,
    indices=train_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=target_seq_len
)

val_dataset = TransformerDataset(
    data=data_tensor,
    indices=val_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=target_seq_len
)

# **Create Train & Validation Dataloaders**
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# **Test Data Load**
src, trg, trg_y = next(iter(train_loader))
print(f"Train data shapes: {src.shape}, {trg.shape}, {trg_y.shape}")

# 3. Initialize Model
model = TimeSeriesTransformer(
    input_size=src.shape[2],
    dec_seq_len=dec_seq_len,
    batch_first=True,
)

# **Select device: MPS (Mac) or CPU**
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)
print(f"Model initialized on: {device}")


# 4. Define Loss & Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00007)

# Gradient Clipping
clip_value = 1.0


# 5. Training Loop
num_epochs = 20  
save_path = "transformer_lorenz96(epoch=20, lr=0.00007).pth"

print("\nStarting training...")

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for src, trg, trg_y in train_loader:
        # Move data to device
        src, trg, trg_y = src.to(device), trg.to(device), trg_y.to(device)

        optimizer.zero_grad()  # Reset gradients

        # Create masks
        src_mask = utils.generate_square_subsequent_mask(target_seq_len, enc_seq_len).to(device)
        tgt_mask = utils.generate_square_subsequent_mask(target_seq_len, target_seq_len).to(device)

        # Forward pass
        output = model(src, trg, src_mask=src_mask, tgt_mask=tgt_mask)

        # Ensure output shape matches target shape
        if output.shape != trg_y.shape:
            print(f"Warning: output shape {output.shape} does not match trg_y {trg_y.shape}, adjusting shape...")
            output = output[:, : trg_y.shape[1], :]

        # Compute loss
        loss = criterion(output, trg_y)
        loss.backward()  # Backpropagation

        # Gradient Clipping (to prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()  # Update weights
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # **Validation Loop (No Gradient Update)**
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for src, trg, trg_y in val_loader:
            src, trg, trg_y = src.to(device), trg.to(device), trg_y.to(device)

            src_mask = utils.generate_square_subsequent_mask(target_seq_len, enc_seq_len).to(device)
            tgt_mask = utils.generate_square_subsequent_mask(target_seq_len, target_seq_len).to(device)

            output = model(src, trg, src_mask=src_mask, tgt_mask=tgt_mask)

            if output.shape != trg_y.shape:
                output = output[:, :trg_y.shape[1], :]
            # Rescale output and target
            val_loss = criterion(output, trg_y)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

print("\nTraining complete!")


# 6. Save the Trained Model
torch.save(model.state_dict(), save_path)
print(f"Model saved as {save_path}")


# 7. Quick Validation
print("\nRunning quick validation...")
model.eval()

with torch.no_grad():
    src, trg, trg_y = val_dataset[0]  # Get the first sample
    src, trg = src.unsqueeze(0).to(device), trg.unsqueeze(0).to(device)  # Add batch dimension

    prediction = model(src, trg)  # Generate prediction

print(f"Validation complete, output shape: {prediction.shape}")  # Expected (1, 20, 3)

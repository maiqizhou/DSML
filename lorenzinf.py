import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transformer_timeseries import TimeSeriesTransformer
import utils

def run_continuous_inference(model_path, initial_sequence, total_pred_steps, enc_seq_len, device="mps"):
    """
    Runs continuous inference by using the model's predictions as inputs for the next step.
    """

    model = TimeSeriesTransformer(
        input_size=3,        
        dec_seq_len=1,      
        batch_first=True
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()

    if isinstance(initial_sequence, np.ndarray):
        initial_sequence = torch.tensor(initial_sequence, dtype=torch.float32)

    input_sequence = initial_sequence.unsqueeze(0).to(device) 

    predicted_trajectory = []

    with torch.no_grad():
        for _ in range(total_pred_steps):
            decoder_input = input_sequence[:, -1:, :]
            src_mask = utils.generate_square_subsequent_mask(decoder_input.shape[1], input_sequence.shape[1]).to(device)
            tgt_mask = utils.generate_square_subsequent_mask(decoder_input.shape[1], decoder_input.shape[1]).to(device)
            output = model(input_sequence, decoder_input, src_mask=src_mask, tgt_mask=tgt_mask)
            next_pred = output[:, -1:, :]
            predicted_trajectory.append(next_pred.cpu().numpy())  
            input_sequence = torch.cat((input_sequence[:, 1:, :], next_pred), dim=1)

    predicted_trajectory = np.concatenate(predicted_trajectory, axis=1).squeeze(0)  
    return predicted_trajectory

# Load test data
test_data_path = "finalProject/DSML/lorenz63_test.npy"
test_data = np.load(test_data_path)

# Extract last `enc_seq_len` steps from test data as initial input
enc_seq_len = 30  
total_pred_steps = 1000  
initial_sequence = test_data[-enc_seq_len:]  

initial_tensor = torch.tensor(initial_sequence, dtype=torch.float32)

# Run continuous inference
model_path = "fine_tuned_transformer.pth"
predicted_output = run_continuous_inference(model_path, initial_tensor, total_pred_steps, enc_seq_len)

# 🔹 3D Visualization with Gradient Color
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Create color map for time evolution
colors = np.linspace(0, 1, total_pred_steps)

# Plot true trajectory
ax.plot(test_data[:, 0], test_data[:, 1], test_data[:, 2], color='gray', linestyle='-', alpha=0.5, label="True Trajectory")

# Plot predicted trajectory with gradient color
for i in range(total_pred_steps - 1):
    ax.plot(predicted_output[i:i+2, 0], 
            predicted_output[i:i+2, 1], 
            predicted_output[i:i+2, 2], 
            color=plt.cm.plasma(colors[i]), linewidth=2)

# Labels and title
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("Lorenz-63 Continuous Prediction with Gradient Color")
ax.legend()
plt.show()


# 🔹 2D Subplot for Each Variable
time_range = np.arange(total_pred_steps)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

var_names = ["x", "y", "z"]
for i, ax in enumerate(axes):
    ax.plot(time_range, predicted_output[:, i], label=f"Predicted {var_names[i]}", linestyle="dashed", color="blue")
    ax.plot(time_range, test_data[enc_seq_len:enc_seq_len + total_pred_steps, i], label=f"True {var_names[i]}", color="red")
    ax.legend()
    ax.set_ylabel(var_names[i])

axes[-1].set_xlabel("Time Step")
plt.suptitle("Lorenz-63 Prediction vs. Ground Truth (x, y, z)")
plt.show()

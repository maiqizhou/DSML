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
        batch_first=True,
        num_predicted_features = 3
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

# Extract first `enc_seq_len` steps from test data as initial input
enc_seq_len = 30  
total_pred_steps = len(test_data) - enc_seq_len
initial_sequence = test_data[:enc_seq_len]  

initial_tensor = torch.tensor(initial_sequence, dtype=torch.float32)

# Run continuous inference
model_path = "transformer_lorenz63(epoch=10, lr=0.0001).pth"
save_path = "generated_lorenz63(epoch=10, lr=0.0001).npy"
predicted_output = run_continuous_inference(model_path, initial_tensor, total_pred_steps, enc_seq_len)

# Save generated time series for future use

if isinstance(predicted_output, torch.Tensor):
    np.save(save_path, predicted_output.cpu().numpy()) 
else:
    np.save(save_path, predicted_output)
print(f"Generated time series saved at: {save_path}")
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot true trajectory in gray
ax.plot(test_data[:, 0], test_data[:, 1], test_data[:, 2], color='gray', linestyle='-', alpha=0.5, label="True Trajectory")

# Plot predicted trajectory in a single color (e.g., blue)
ax.plot(predicted_output[:, 0], predicted_output[:, 1], predicted_output[:, 2], color='blue', linewidth=2, label="Predicted Trajectory")

# Mark start and end points with small markers
ax.scatter(predicted_output[0, 0], predicted_output[0, 1], predicted_output[0, 2], 
           color="red", s=30, label="Start Point", edgecolors="black")  
ax.scatter(predicted_output[-1, 0], predicted_output[-1, 1], predicted_output[-1, 2], 
           color="green", s=30, label="End Point", edgecolors="black")  

# Labels and title
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("Lorenz-63 Continuous Prediction")
ax.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load true Lorenz-63 data (test_data) and predicted data (lorenz63_data)
test_data_path = "lorenz63_test.npy"
generated_data_path = "generated_lorenz63(epoch=10, lr=0.0001).npy"

test_data = np.load(test_data_path)  # Load true Lorenz-63 trajectory
lorenz63_data = np.load(generated_data_path)  # Load predicted trajectory
initial_steps = test_data[:30, :] 
generated_series = np.concatenate((initial_steps, lorenz63_data), axis=0)  # Along the time axis
print(f"Loaded true data: {test_data.shape}, Generated data: {lorenz63_data.shape}")
# 3D Trajectory Visualization
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot true trajectory in gray (semi-transparent for better visibility)
ax.plot(test_data[:, 0], test_data[:, 1], test_data[:, 2], 
        color='gray', linestyle='-', alpha=0.5, label="True Trajectory")

# Plot predicted trajectory in blue with a thin line
ax.plot(lorenz63_data[:, 0], lorenz63_data[:, 1], lorenz63_data[:, 2], 
        color='purple', linewidth=0.2, label="Predicted Trajectory")

# Mark start and end points with small markers
ax.scatter(lorenz63_data[0, 0], lorenz63_data[0, 1], lorenz63_data[0, 2], 
           color="red", s=30, label="Start Point", edgecolors="black")  # Start point (Red)
ax.scatter(lorenz63_data[-1, 0], lorenz63_data[-1, 1], lorenz63_data[-1, 2], 
           color="green", s=30, label="End Point", edgecolors="black")  # End point (Green)

# Set labels and title
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("Lorenz-63: True vs. Predicted Trajectory(T=the length of the test set)")
ax.legend()
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Load Lorenz-96 data (assuming shape: [time_steps, num_dimensions])
# lorenz96_data_path = "generated_lorenz96(epoch=20, lr=0.00007).npy"
# lorenz96_data = np.load(lorenz96_data_path)  # Shape: (time_steps, 20) if using 20D system

# # Select three dimensions for 3D plotting (e.g., X1, X2, X3)
# X1, X2, X3 = lorenz96_data[:, 0], lorenz96_data[:, 1], lorenz96_data[:, 2]

# # 🔹 3D Trajectory Plot
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')

# # Plot Lorenz-96 trajectory using X1, X2, and X3
# ax.plot(X1, X2, X3, color='blue', linewidth=0.1, label="Lorenz-96 Trajectory")

# # Mark start and end points
# ax.scatter(X1[0], X2[0], X3[0], color="red", s=30, label="Start Point", edgecolors="black")
# ax.scatter(X1[-1], X2[-1], X3[-1], color="green", s=30, label="End Point", edgecolors="black")

# # Set labels and title
# ax.set_xlabel("X1")
# ax.set_ylabel("X2")
# ax.set_zlabel("X3")
# ax.set_title("Lorenz-96: 3D Projection (X1, X2, X3)")
# ax.legend()
# plt.show()

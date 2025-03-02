# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Load true Lorenz-63 data (test_data) and predicted data (lorenz63_data)
# test_data_path = "finalProject/DSML/lorenz63_test.npy"
# generated_data_path = "generated_lorenz63(epoch=20, lr=0.00007).npy"

# test_data = np.load(test_data_path)  # Load true Lorenz-63 trajectory
# lorenz63_data = np.load(generated_data_path)  # Load predicted trajectory
# initial_steps = test_data[:30, :] 
# assert len(lorenz63_data.shape) == 2, "Expected shape (time_steps, 20) but got {}".format(lorenz63_data.shape)
# generated_series = np.concatenate((initial_steps, lorenz63_data), axis=0)  # Along the time axis
# print(f"Loaded true data: {test_data.shape}, Generated data: {lorenz63_data.shape}")
# # 3D Trajectory Visualization
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')

# # Plot true trajectory in gray (semi-transparent for better visibility)
# ax.plot(test_data[:, 0], test_data[:, 1], test_data[:, 2], 
#         color='gray', linestyle='-', alpha=0.5, label="True Trajectory")

# # Plot predicted trajectory in blue with a thin line
# ax.plot(lorenz63_data[:, 0], lorenz63_data[:, 1], lorenz63_data[:, 2], 
#         color='yellow', linewidth=0.2, label="Predicted Trajectory")

# # Mark start and end points with small markers
# ax.scatter(lorenz63_data[0, 0], lorenz63_data[0, 1], lorenz63_data[0, 2], 
#            color="red", s=30, label="Start Point", edgecolors="black")  # Start point (Red)
# ax.scatter(lorenz63_data[-1, 0], lorenz63_data[-1, 1], lorenz63_data[-1, 2], 
#            color="green", s=30, label="End Point", edgecolors="black")  # End point (Green)

# # Set labels and title
# ax.set_xlabel("X-axis")
# ax.set_ylabel("Y-axis")
# ax.set_zlabel("Z-axis")
# ax.set_title("Lorenz-63: True vs. Predicted Trajectory(T=the length of the test set)")
# ax.legend()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load true Lorenz-63 test data
test_data_path = "finalProject/DSML/lorenz96_test.npy"
test_data = np.load(test_data_path)  # Shape: (time_steps, 3) for Lorenz-63

# Load generated Lorenz-96 data
generated_data_path = "generated_lorenz96(epoch=20, lr=0.00007).npy"
lorenz96_data = np.load(generated_data_path)  # Shape: (time_steps, 20)
print(lorenz96_data.shape)
print(test_data.shape)

# Ensure both datasets have the same number of time steps for proper comparison
time_steps = min(test_data.shape[0], lorenz96_data.shape[0])  # Choose the shortest length
test_data = test_data[:time_steps]
lorenz96_data = lorenz96_data[:time_steps]

print(f"Loaded test data: {test_data.shape}, Generated Lorenz-96 data: {lorenz96_data.shape}")

fig, axes = plt.subplots(5, 4, figsize=(12, 10), sharex=True, sharey=True)  # 5 rows, 4 columns

time_range = np.arange(time_steps)

for i, ax in enumerate(axes.flat):
    ax.plot(time_range, lorenz96_data[:, i], label=f"X{i+1}", color='blue')
    ax.legend()
    ax.set_ylabel(f"X{i+1}")

axes[-1, -1].set_xlabel("Time Step")
plt.suptitle("Lorenz-96: Evolution of 20 Dimensions Over Time")
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 6))
sns.heatmap(lorenz96_data.T, cmap="coolwarm", cbar=True, xticklabels=100, yticklabels=1)

plt.xlabel("Time Step")
plt.ylabel("Dimension Index")
plt.title("Lorenz-96: Heatmap of 20 Variables Over Time")
plt.show()

import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from transformer_timeseries import TimeSeriesTransformer

SMOOTHING_SIGMA = 20
# Select device: Apple MPS, CUDA, or CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")


# compute power spectra distances and average across all dimensions
def power_spectrum_error(x_gen, x_true):
    pse_errors_per_dim = power_spectrum_error_per_dim(x_gen, x_true)
    # Convert list of tensor distances to numpy array and compute mean
    return torch.tensor([pe.item() for pe in pse_errors_per_dim]).mean().item()

def compute_power_spectrum(x):
    """
    Computes the power spectrum of a time series using FFT in PyTorch.
    """
    # Use torch.fft.rfft (falling back to CPU if needed)
    x_cpu = x.to("cpu")
    fft_real = torch.fft.rfft(x_cpu, dim=1)
    ps = torch.abs(fft_real) ** 2
    ps = ps / (torch.sum(ps, dim=1, keepdim=True) + 1e-8)
    return ps  # stays as a torch.Tensor

def get_average_spectrum(x):
    """
    Normalize individual trajectories and compute their average power spectrum.
    """
    # Normalize: subtract mean and divide by std
    x_ = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
    spectrum = compute_power_spectrum(x_)
    # Normalize the spectrum to sum to 1
    return spectrum / (spectrum.sum(dim=1, keepdim=True) + 1e-8)

def power_spectrum_error_per_dim(x_gen, x_true):
    """
    Computes Power Spectrum Distance (PSD) for each variable.
    Both x_gen and x_true are expected to be torch.Tensors with shape [batch, time, features].
    """
    assert x_true.shape[1] == x_gen.shape[1]
    assert x_true.shape[2] == x_gen.shape[2]
    dim_x = x_gen.shape[2]
    pse_per_dim = []
    for dim in range(dim_x):
        # Get the average spectrum for the current dimension
        spectrum_true = get_average_spectrum(x_true[:, :, dim])
        spectrum_gen  = get_average_spectrum(x_gen[:, :, dim])
        hd = hellinger_distance(spectrum_true, spectrum_gen)
        pse_per_dim.append(hd)
    return pse_per_dim

def hellinger_distance(p, q):
    """
    Computes the Hellinger distance between two probability distributions.
    Both p and q are torch.Tensors.
    """
    diff = torch.sqrt(p) - torch.sqrt(q)
    return 1 / torch.sqrt(torch.tensor(2.0)) * torch.sqrt(torch.sum(diff ** 2, dim=1))

# functions for smoothing power spectra with a Gaussian kernel using SciPy
def kernel_smoothen(data, kernel_sigma=1):
    """
    Smoothens data with a Gaussian kernel.
    The input 'data' is expected to be a NumPy array.
    This function returns a NumPy array.
    """
    kernel = get_kernel(kernel_sigma)
    data_final = data.copy()
    data_conv = np.convolve(data[:], kernel, mode="same")
    return data_conv

def gauss(x, sigma=1):
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-0.5 * (x / sigma) ** 2)

def get_kernel(sigma):
    size = int(sigma * 10 + 1)
    kernel = list(range(size))
    kernel = [float(k) - int(size / 2) for k in kernel]
    kernel = [gauss(k, sigma) for k in kernel]
    kernel = [k / np.sum(kernel) for k in kernel]
    return np.array(kernel)




# Load the generated time series
generated_series = np.load("/Users/maggie/Documents/Semester4/DSML/generated_lorenz63(epoch=20, lr=0.00007).npy")

# Load the true test set time series
true_series = np.load("finalProject/DSML/lorenz63_test.npy")


# Convert to tensors
generated_series = torch.tensor(generated_series, dtype=torch.float32)
true_series = torch.tensor(true_series, dtype=torch.float32)
# Ensure both have the same shape for comparison
generated_series = generated_series.unsqueeze(0)
true_series = true_series.unsqueeze(0)
initial_steps = true_series[:, :30, :]
generated_series = torch.cat((initial_steps, generated_series), dim=1)

# Ensure both have the same shape for comparison
print("Generated Series Shape:", generated_series.shape)
print("True Series Shape:", true_series.shape)

ps_distance = power_spectrum_error(generated_series, true_series)
print(f"\nPower Spectrum Distance: {ps_distance:.6f}")
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from transformer_timeseries import TimeSeriesTransformer
import utils

# Select device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

SMOOTHING_SIGMA = 20  # Gaussian smoothing parameter


def compute_power_spectrum(x):
    """
    Computes the power spectrum of a time series using FFT in PyTorch.
    Falls back to CPU since FFT is not supported on MPS.
    """
    x_cpu = x.to("cpu")  
    fft_real = torch.fft.rfft(x_cpu, dim=1)  
    ps = torch.abs(fft_real) ** 2  
    ps = ps / ps.sum(dim=1, keepdim=True)  
    return ps.to(device)


def smooth_spectrum(ps, sigma=SMOOTHING_SIGMA):
    """
    Applies Gaussian smoothing to the power spectrum.
    """
    ps_np = ps.cpu().numpy()  
    smoothed_ps = np.apply_along_axis(lambda x: gaussian_filter1d(x, sigma), axis=1, arr=ps_np)
    return torch.tensor(smoothed_ps, dtype=torch.float32, device=device)


def hellinger_distance(p, q):
    """
    Computes the Hellinger distance between two probability distributions.
    """
    return torch.sqrt(torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2, dim=1)) / torch.sqrt(torch.tensor(2.0, device=device))


def power_spectrum_error_per_dim(x_gen, x_true):
    """
    Computes Power Spectrum Distance (PSD) for each variable (x, y, z).
    """
    assert x_true.shape[1] == x_gen.shape[1]
    assert x_true.shape[2] == x_gen.shape[2]

    dim_x = x_gen.shape[2]
    pse_per_dim = []

    for dim in range(dim_x):
        ps_true = compute_power_spectrum(x_true[:, :, dim])  
        ps_pred = compute_power_spectrum(x_gen[:, :, dim])  

        # Apply Gaussian smoothing
        ps_true = smooth_spectrum(ps_true)
        ps_pred = smooth_spectrum(ps_pred)

        # Compute Hellinger distance
        hd = hellinger_distance(ps_true, ps_pred).mean()
        pse_per_dim.append(hd)

    return torch.stack(pse_per_dim).mean().to(device)


def generate_time_series(model, initial_condition, total_pred_steps):
    """
    Generates a long time series (T steps) from a given initial condition.
    """
    model.eval()
    generated_series = []
    with torch.no_grad():
        input_sequence = initial_condition

        for _ in range(total_pred_steps):
            decoder_input = input_sequence[:, -1:, :]
            output = model(input_sequence, decoder_input)
            next_pred = output[:, -1:, :]
            generated_series.append(next_pred.cpu().numpy())

            input_sequence = torch.cat((input_sequence[:, 1:, :], next_pred), dim=1)

    return np.concatenate(generated_series, axis=1).squeeze(0)


def run_psd_analysis(model_path, test_data_path, enc_seq_len, total_pred_steps):
    """
    Runs inference to generate a time series and compares power spectra between
    the ground truth and predicted series.
    """
    print("Loading model...")
    model = TimeSeriesTransformer(input_size=3, dec_seq_len=1, batch_first=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    print("Loading test data...")
    test_data = np.load(test_data_path)

    # Randomly sample an initial condition from test data
    random_start_idx = np.random.randint(0, len(test_data) - enc_seq_len - total_pred_steps)
    initial_sequence = test_data[random_start_idx:random_start_idx + enc_seq_len]
    
    initial_tensor = torch.tensor(initial_sequence, dtype=torch.float32).unsqueeze(0).to(device)

    # Generate long time series
    print("Generating time series...")
    predicted_output = generate_time_series(model, initial_tensor, total_pred_steps)

    # Compute power spectra
    print("Computing power spectrum distance...")
    ps_true = compute_power_spectrum(torch.tensor(test_data[random_start_idx:random_start_idx + total_pred_steps], dtype=torch.float32).unsqueeze(0))
    ps_pred = compute_power_spectrum(torch.tensor(predicted_output, dtype=torch.float32).unsqueeze(0))
    
    print(f"Power Spectrum Shape (True): {ps_true.shape}")
    print(f"Power Spectrum Shape (Pred): {ps_pred.shape}")

    # Compute Hellinger distance
    ps_distance = hellinger_distance(ps_true, ps_pred).mean().item()
    print(f"Power Spectrum Distance: {ps_distance:.6f}")

    # Plot power spectra
    plt.figure(figsize=(8, 5))
    plt.plot(ps_true.cpu().numpy().squeeze(), label="True Power Spectrum", color="red")
    plt.plot(ps_pred.cpu().numpy().squeeze(), label="Predicted Power Spectrum", color="blue", linestyle="dashed")
    plt.legend()
    plt.xlabel("Frequency")
    plt.ylabel("Power")
    plt.title("Power Spectrum Comparison")
    plt.show()


# Run the power spectrum analysis
run_psd_analysis("fine_tuned_transformer.pth", "finalProject/DSML/lorenz63_test.npy", enc_seq_len=30, total_pred_steps=200)

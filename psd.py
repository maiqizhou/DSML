import torch
import torch.nn as nn
import torch.nn.functional as F

SMOOTHING_SIGMA = 20  

# Ensure calculations happen on the correct device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def compute_power_spectrum(x):
    """
    Computes the power spectrum of a given time series batch using FFT in PyTorch.
    Falls back to CPU since FFT is not supported on MPS.
    """
    x_cpu = x.to("cpu")  
    fft_real = torch.fft.rfft(x_cpu, dim=1)  
    ps = torch.abs(fft_real) ** 2  
    ps = ps / ps.sum(dim=1, keepdim=True) 

    return ps.to("mps") 

def hellinger_distance(p, q):
    """
    Computes the Hellinger distance between two probability distributions using PyTorch.
    """
    return torch.sqrt(torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2, dim=1)) / torch.sqrt(torch.tensor(2.0, device=device))

def power_spectrum_error_per_dim(x_gen, x_true):
    """
    Computes Power Spectrum Distance for each variable (x, y, z).
    """
    assert x_true.shape[1] == x_gen.shape[1]  
    assert x_true.shape[2] == x_gen.shape[2]  
    
    dim_x = x_gen.shape[2]  
    pse_per_dim = []

    for dim in range(dim_x):
        ps_true = compute_power_spectrum(x_true[:, :, dim]) 
        ps_pred = compute_power_spectrum(x_gen[:, :, dim])  
        hd = hellinger_distance(ps_true, ps_pred).mean()  
        pse_per_dim.append(hd)

    return torch.stack(pse_per_dim).mean().to(device)  

def power_spectrum_loss(y_true, y_pred):
    """
    Computes the Power Spectrum Distance (PSD) Loss by averaging Hellinger distances 
    over all three variables (x, y, z).
    """
    return power_spectrum_error_per_dim(y_pred, y_true)

def correlation_loss(y_true, y_pred):
    """
    Computes correlation loss for each of the 3 predicted variables (x, y, z).
    """
    y_true_mean = y_true.mean(dim=1, keepdim=True)
    y_pred_mean = y_pred.mean(dim=1, keepdim=True)

    y_true_std = y_true.std(dim=1, keepdim=True) + 1e-6
    y_pred_std = y_pred.std(dim=1, keepdim=True) + 1e-6

    corr = ((y_true - y_true_mean) * (y_pred - y_pred_mean)).mean(dim=1) / (y_true_std * y_pred_std)
    return (1 - corr.mean()).to(device)  

def combined_loss(y_true, y_pred, alpha=1.0, beta=0.5, gamma=0.1):
    """
    Computes the final loss function combining MSE, PSD, and correlation loss.
    """
    loss_mse = nn.MSELoss()(y_pred, y_true)
    loss_psd = power_spectrum_loss(y_true, y_pred)
    # loss_corr = correlation_loss(y_true, y_pred)
    total_loss = alpha * loss_mse + beta * loss_psd 
    return total_loss.to(device)  

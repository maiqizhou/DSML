# Lorenz System Time Series Forecasting with Transformer
This project implements a Transformer model for time series forecasting on two classic chaotic systems: Lorenz-63 and Lorenz-96.

### Scripts Overview
train.py: Trains the Transformer model on the Lorenz datasets.

psd.py: Evaluates model performance by comparing ground truth and generated data using Power Spectral Density (PSD).

lorenzinf.py: Performs inference using the trained model.

plot.py: Visualizes the results.

### Usage
Train the model by running train.py.

Generate predictions using lorenzinf.py.

Visualize results with plot.py.

Evaluate performance using psd.py.

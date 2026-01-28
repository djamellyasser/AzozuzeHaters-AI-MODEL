import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(dataset_path):
    """
    Load the pre-processed numpy arrays for the MIT-CHB dataset.
    """
    print("Loading dataset...")
    X = np.load(os.path.join(dataset_path, 'signal_samples.npy'))
    y = np.load(os.path.join(dataset_path, 'is_sz.npy'))
    print(f"Original X shape: {X.shape}")
    print(f"Original y shape: {y.shape}")
    return X, y

def preprocess_data(X, y):
    """
    Transpose signals, convert labels to binary, split, and standardize.
    """
    # Transpose from (N, Channels, Time) to (N, Time, Channels)
    X = np.transpose(X, (0, 2, 1))
    print(f"Transposed X shape: {X.shape}")

    # Binary labels
    y_binary = (y > 0.5).astype(int)

    # Split into Training and Test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    # Standardization
    samples, time_steps, channels = X_train.shape
    X_train_flat = X_train.reshape(-1, channels)
    X_test_flat = X_test.reshape(-1, channels)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_flat).reshape(samples, time_steps, channels)
    X_test = scaler.transform(X_test_flat).reshape(X_test.shape[0], time_steps, channels)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler

# Hybrid CNN-LSTM for Seizure Detection

This project implements a state-of-the-art hybrid deep learning model for detecting seizures in EEG signals using the MIT-CHB dataset. The architecture leverages 1D Convolutional Neural Networks (CNN) for spatial feature extraction and Bidirectional Long Short-Term Memory (Bi-LSTM) networks for capturing temporal dynamics.

## 🚀 Overview

Seizure detection is a critical task in neurological monitoring. This model processes raw EEG signals, extracts high-frequency features through three convolutional blocks, and summarizes temporal progression using a Bidirectional LSTM layer. It achieves high accuracy and robust performance on binary classification (Seizure vs. Non-Seizure).

## 📊 Model Architecture

The model consists of:
1.  **CNN blocks:** Three blocks of Conv1D, BatchNormalization, ReLU activation, MaxPooling, and Dropout.
    -   Block 1: 64 filters
    -   Block 2: 128 filters
    -   Block 3: 256 filters
2.  **RNN layer:** A Bidirectional LSTM with 128 units to capture context from both directions.
3.  **Global Average Pooling:** Summarizes the sequence dimension into a single feature vector.
4.  **Dense Head:** A fully connected layer with 128 units followed by a Sigmoid output for binary classification.

## 📂 Project Structure

```text
seizure_detection_model/
├── main.py              # Main orchestration script
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
├── data/                 # Directory for dataset files
├── output/               # Directory for saved models and plots
└── src/                  # Source code package
    ├── __init__.py
    ├── data_preprocessing.py
    ├── model_arch.py
    ├── training.py
    └── evaluation.py
```

-   `main.py`: The entry point for the entire pipeline (loading, training, evaluation).
-   `src/data_preprocessing.py`: Logic for data loading and cleaning.
-   `src/model_arch.py`: The Hybrid CNN-LSTM model definition.
-   `src/training.py`: Training routines and callback configurations.
-   `src/evaluation.py`: Performance visualization and metric calculation.

## 🛠️ Installation

1.  Clone the repository or download the files.
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 📈 Usage

1.  Prepare your dataset (ensure `signal_samples.npy` and `is_sz.npy` are available).
2.  Update the `DATASET_PATH` in `main.py` to point to your data directory.
3.  Run the training pipeline:
    ```bash
    python main.py
    ```

## 📝 Performance Metrics

The model tracks:
-   **Binary Accuracy**
-   **Area Under the ROC Curve (AUC)**
-   **Precision, Recall, and F1-Score**

Calculated metrics are saved and visualized via `evaluate.py`.

## 🧠 Dataset Information

This implementation is designed for the **MIT-CHB Processed** dataset, where EEG signals are expected in the shape `(Samples, Channels, Time)`. The preprocessing step automatically transposes this to `(Batch, Time, Channels)` for efficient 1D-Convolution processing.

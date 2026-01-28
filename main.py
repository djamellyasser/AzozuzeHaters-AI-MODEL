import os
import tensorflow as tf
from src.data_preprocessing import load_data, preprocess_data
from src.model_arch import build_hybrid_model, compile_model
from src.training import train_model
from src.evaluation import evaluate_model, plot_results

# Configuration
DATASET_PATH = '/kaggle/input/mit-chb-processed/' # Update this to your local path
OUTPUT_DIR = 'output'

def main():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Check for GPU
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # 1. Data Loading and Preprocessing
    if not os.path.exists(DATASET_PATH):
        print(f"Warning: Dataset path {DATASET_PATH} not found. Please update DATASET_PATH in main.py")
        
    X_raw, y_raw = load_data(DATASET_PATH)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X_raw, y_raw)

    # 2. Build and Compile Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_hybrid_model(input_shape)
    model = compile_model(model)
    model.summary()

    # 3. Training
    best_model_path = os.path.join(OUTPUT_DIR, 'best_seizure_model.keras')
    history = train_model(model, X_train, y_train, model_save_path=best_model_path)

    # 4. Evaluation
    y_test_eval, y_pred_eval = evaluate_model(model, X_test, y_test, model_path=best_model_path)
    plot_results(history, y_test_eval, y_pred_eval)

if __name__ == "__main__":
    main()

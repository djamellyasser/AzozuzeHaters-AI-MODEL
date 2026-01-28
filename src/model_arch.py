import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, BatchNormalization, 
    Activation, Bidirectional, LSTM, Dense, Dropout, 
    GlobalAveragePooling1D
)

def build_hybrid_model(input_shape):
    """
    Builds a Hybrid 1D-CNN + Bi-LSTM model for EEG seizure detection.
    """
    inputs = Input(shape=input_shape)

    # --- Feature Extraction (CNN Block) ---
    # Block 1
    x = Conv1D(filters=64, kernel_size=5, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # Block 2
    x = Conv1D(filters=128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # Block 3
    x = Conv1D(filters=256, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # --- Temporal Dynamics (RNN Block) ---
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.4)(x)
    
    # Global Average Pooling to summarize time dimension
    x = GlobalAveragePooling1D()(x)

    # --- Classification Head ---
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def compile_model(model, learning_rate=0.001):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

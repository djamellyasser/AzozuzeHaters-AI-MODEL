from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def train_model(model, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, model_save_path='best_model.keras'):
    """
    Trains the model with predefined callbacks.
    """
    checkpoint = ModelCheckpoint(
        model_save_path, 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max', 
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=0.00001, 
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True, 
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, reduce_lr, early_stop],
        verbose=1
    )
    return history

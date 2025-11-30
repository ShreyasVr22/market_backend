
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle
import os

class LSTMPredictor:
    def __init__(self, lookback=30):
        self.lookback = lookback
        self.model = None
        self.scaler = None
    
    def build_model(self, input_shape=(30, 1)):
        """Build LSTM model"""
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return self.model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """Train LSTM model"""
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_split=0.2
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def save_model(self, filepath):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"✓ Model saved: {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
        print(f"✓ Model loaded: {filepath}")

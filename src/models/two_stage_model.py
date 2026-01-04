"""
Two-Stage Bot Detection Model Architecture
Stage 1: Fast Logistic Regression Filter
Stage 2: Deep GRU-based Sequence Model
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Tuple, Dict, Optional
import joblib


class FastFilterModel:
    """Stage 1: Fast Logistic Regression Filter for quick screening."""

    def __init__(self, random_state: int = 42, max_iter: int = 1000):
        """Initialize fast filter model."""
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            class_weight='balanced',
            solver='lbfgs'
        )
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray, class_weights: Optional[Dict] = None) -> Dict:
        """
        Train the fast filter model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            class_weights: Optional class weights dictionary
        
        Returns:
            Training metrics dictionary
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on training data
        y_pred = self.model.predict(X_train)
        y_pred_proba = self.model.predict_proba(X_train)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_train, y_pred),
            'precision': precision_score(y_train, y_pred, zero_division=0),
            'recall': recall_score(y_train, y_pred, zero_division=0),
            'f1': f1_score(y_train, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_train, y_pred_proba)
        }
        
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict bot probability."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)[:, 1]

    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict bot class with threshold."""
        proba = self.predict(X)
        return (proba >= threshold).astype(int)

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        joblib.dump(self.model, filepath)

    def load(self, filepath: str) -> None:
        """Load model from disk."""
        self.model = joblib.load(filepath)
        self.is_trained = True


class GRUSequenceModel(nn.Module):
    """Stage 2: Deep GRU-based sequence model for sophisticated bot detection."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize GRU sequence model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            num_layers: Number of GRU layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional GRU
        """
        super(GRUSequenceModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Fully connected layers
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(gru_output_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            Output tensor of shape (batch_size, 1) with bot probability
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Use last output from GRU
        last_output = gru_out[:, -1, :]
        
        # Fully connected layers
        x = self.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        
        return x


class SequenceModelTrainer:
    """Trainer for the GRU sequence model."""

    def __init__(
        self,
        model: GRUSequenceModel,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """Initialize trainer."""
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCELoss()
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def train_epoch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int = 32
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(batch_X).to(self.device)
            y_tensor = torch.FloatTensor(batch_y).reshape(-1, 1).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches

    def validate(self, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X_val).to(self.device)
        y_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            
            # Calculate accuracy
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y_tensor).float().mean().item()
        
        return loss.item(), accuracy

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10
    ) -> Dict:
        """
        Train the model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
        
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(X_train, y_train, batch_size)
            
            # Validate
            val_loss, val_acc = self.validate(X_val, y_val)
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict bot probability."""
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
        
        return outputs.cpu().numpy().flatten()

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        """Load model from disk."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))


class TwoStageDetector:
    """Complete two-stage bot detection system."""

    def __init__(self, device: str = 'cpu'):
        """Initialize two-stage detector."""
        self.fast_filter = FastFilterModel()
        self.sequence_model = None
        self.sequence_trainer = None
        self.device = device
        self.filter_threshold = 0.5
        self.sequence_threshold = 0.5

    def train_stage1(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train stage 1 (fast filter)."""
        print("Training Stage 1: Fast Filter...")
        metrics = self.fast_filter.train(X_train, y_train)
        print(f"Stage 1 Metrics: {metrics}")
        return metrics

    def train_stage2(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_size: int,
        epochs: int = 50
    ) -> Dict:
        """Train stage 2 (sequence model)."""
        print("Training Stage 2: Sequence Model...")
        
        # Initialize model
        self.sequence_model = GRUSequenceModel(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            bidirectional=True
        )
        
        self.sequence_trainer = SequenceModelTrainer(
            self.sequence_model,
            device=self.device,
            learning_rate=0.001
        )
        
        # Reshape data for sequence model (add sequence dimension)
        X_train_seq = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val_seq = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        
        history = self.sequence_trainer.train(
            X_train_seq, y_train,
            X_val_seq, y_val,
            epochs=epochs,
            batch_size=32,
            early_stopping_patience=10
        )
        
        print("Stage 2 Training Complete!")
        return history

    def predict(self, X: np.ndarray, use_both_stages: bool = True) -> Dict:
        """
        Predict bot probability using one or both stages.
        
        Args:
            X: Input features
            use_both_stages: If True, use both stages; if False, only use stage 2
        
        Returns:
            Dictionary with predictions and confidence scores
        """
        if use_both_stages:
            # Stage 1: Fast filter
            stage1_proba = self.fast_filter.predict(X)
            
            # Stage 2: Sequence model (only for uncertain cases)
            X_seq = X.reshape(X.shape[0], 1, X.shape[1])
            stage2_proba = self.sequence_trainer.predict(X_seq)
            
            # Combine predictions (weighted average)
            final_proba = 0.3 * stage1_proba + 0.7 * stage2_proba
        else:
            # Only stage 2
            X_seq = X.reshape(X.shape[0], 1, X.shape[1])
            final_proba = self.sequence_trainer.predict(X_seq)
        
        return {
            'bot_probability': final_proba,
            'is_bot': (final_proba >= self.sequence_threshold).astype(int),
            'confidence': np.abs(final_proba - 0.5) * 2  # Confidence score 0-1
        }

    def save(self, stage1_path: str, stage2_path: str) -> None:
        """Save both models."""
        self.fast_filter.save(stage1_path)
        if self.sequence_trainer:
            self.sequence_trainer.save(stage2_path)

    def load(self, stage1_path: str, stage2_path: str, input_size: int) -> None:
        """Load both models."""
        self.fast_filter.load(stage1_path)
        
        # Initialize and load sequence model
        self.sequence_model = GRUSequenceModel(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            bidirectional=True
        )
        self.sequence_trainer = SequenceModelTrainer(
            self.sequence_model,
            device=self.device
        )
        self.sequence_trainer.load(stage2_path)


if __name__ == '__main__':
    # Example usage
    print("Two-Stage Bot Detection Model")
    print("=" * 50)
    
    # Create dummy data
    X_train = np.random.randn(100, 20)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.randn(20, 20)
    y_val = np.random.randint(0, 2, 20)
    X_test = np.random.randn(30, 20)
    
    # Initialize detector
    detector = TwoStageDetector(device='cpu')
    
    # Train stage 1
    detector.train_stage1(X_train, y_train)
    
    # Train stage 2
    detector.train_stage2(X_train, y_train, X_val, y_val, input_size=20, epochs=20)
    
    # Predict
    predictions = detector.predict(X_test, use_both_stages=True)
    print(f"\nPredictions shape: {predictions['bot_probability'].shape}")
    print(f"Bot probability (first 5): {predictions['bot_probability'][:5]}")
    print(f"Confidence (first 5): {predictions['confidence'][:5]}")

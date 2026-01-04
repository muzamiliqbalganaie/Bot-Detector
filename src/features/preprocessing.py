"""
Data Preprocessing Module
Handles data cleaning, normalization, and class imbalance mitigation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple, Optional
import joblib


class DataPreprocessor:
    """Preprocesses data for model training."""

    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: 'standard' or 'robust' (robust is better for outliers)
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = RobustScaler()
        
        self.feature_names = None
        self.is_fitted = False

    def fit_scaler(self, X: pd.DataFrame) -> None:
        """Fit the scaler on training data."""
        self.scaler.fit(X)
        self.feature_names = X.columns.tolist()
        self.is_fitted = True

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        return self.scaler.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform data."""
        self.fit_scaler(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data."""
        return self.scaler.inverse_transform(X)

    def save(self, filepath: str) -> None:
        """Save scaler to disk."""
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, filepath)

    def load(self, filepath: str) -> None:
        """Load scaler from disk."""
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']


class DataSplitter:
    """Handles train/validation/test splits."""

    @staticmethod
    def split_data(
        X: pd.DataFrame,
        y: pd.Series,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
        """
        Split data into train, validation, and test sets with stratification.
        
        Args:
            X: Features
            y: Labels
            train_size: Fraction for training
            val_size: Fraction for validation
            test_size: Fraction for testing
            random_state: Random seed
            stratify: Whether to stratify by class
        
        Returns:
            Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None
        )
        
        # Second split: train vs val
        val_ratio = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_temp if stratify else None
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class ImbalanceHandler:
    """Handles class imbalance in training data."""

    @staticmethod
    def apply_smote(
        X: pd.DataFrame,
        y: pd.Series,
        sampling_strategy: float = 0.8,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.
        
        Args:
            X: Features
            y: Labels
            sampling_strategy: Ratio of minority to majority class (0-1)
            random_state: Random seed
        
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=5
        )
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Convert back to DataFrame to preserve column names
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)
        
        return X_resampled, y_resampled

    @staticmethod
    def calculate_class_weights(y: pd.Series) -> dict:
        """
        Calculate class weights for imbalanced data.
        
        Returns:
            Dictionary mapping class labels to weights
        """
        class_counts = y.value_counts()
        total = len(y)
        
        weights = {}
        for class_label, count in class_counts.items():
            # Weight inversely proportional to class frequency
            weights[class_label] = total / (len(class_counts) * count)
        
        return weights


class FeatureSelector:
    """Selects important features for model training."""

    @staticmethod
    def select_by_variance(
        X: pd.DataFrame,
        threshold: float = 0.01
    ) -> pd.DataFrame:
        """
        Remove features with low variance.
        
        Args:
            X: Features
            threshold: Variance threshold
        
        Returns:
            DataFrame with high-variance features only
        """
        variances = X.var()
        selected_features = variances[variances > threshold].index.tolist()
        
        return X[selected_features]

    @staticmethod
    def select_by_correlation(
        X: pd.DataFrame,
        threshold: float = 0.95
    ) -> pd.DataFrame:
        """
        Remove highly correlated features (keep only one from each pair).
        
        Args:
            X: Features
            threshold: Correlation threshold
        
        Returns:
            DataFrame with uncorrelated features
        """
        corr_matrix = X.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        return X.drop(columns=to_drop)


class CompletePipeline:
    """Complete preprocessing pipeline."""

    def __init__(self, scaler_type: str = 'robust', handle_imbalance: bool = True):
        """Initialize complete pipeline."""
        self.preprocessor = DataPreprocessor(scaler_type=scaler_type)
        self.handle_imbalance = handle_imbalance
        self.feature_names = None

    def prepare_data(
        self,
        features_df: pd.DataFrame,
        target_column: str = 'is_bot',
        test_size: float = 0.2,
        val_size: float = 0.15,
        apply_smote: bool = True,
        random_state: int = 42
    ) -> dict:
        """
        Complete data preparation pipeline.
        
        Args:
            features_df: DataFrame with features and target
            target_column: Name of target column
            test_size: Fraction for test set
            val_size: Fraction for validation set
            apply_smote: Whether to apply SMOTE to training data
            random_state: Random seed
        
        Returns:
            Dictionary with train/val/test data and metadata
        """
        # Separate features and target
        X = features_df.drop(columns=[target_column, 'user_id'])
        y = features_df[target_column]
        
        self.feature_names = X.columns.tolist()
        
        # Remove low-variance features
        X = FeatureSelector.select_by_variance(X, threshold=0.01)
        
        # Remove highly correlated features
        X = FeatureSelector.select_by_correlation(X, threshold=0.95)
        
        # Split data
        train_size = 1 - test_size - val_size
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = DataSplitter.split_data(
            X, y,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            stratify=True
        )
        
        # Fit scaler on training data
        self.preprocessor.fit_scaler(X_train)
        
        # Apply SMOTE to training data if requested
        if apply_smote and self.handle_imbalance:
            X_train_resampled, y_train_resampled = ImbalanceHandler.apply_smote(
                X_train, y_train,
                sampling_strategy=0.8,
                random_state=random_state
            )
            X_train_scaled = self.preprocessor.transform(X_train_resampled)
            y_train = y_train_resampled
        else:
            X_train_scaled = self.preprocessor.transform(X_train)
        
        # Scale validation and test data
        X_val_scaled = self.preprocessor.transform(X_val)
        X_test_scaled = self.preprocessor.transform(X_test)
        
        # Calculate class weights
        class_weights = ImbalanceHandler.calculate_class_weights(y_train)
        
        return {
            'X_train': X_train_scaled,
            'y_train': y_train.values,
            'X_val': X_val_scaled,
            'y_val': y_val.values,
            'X_test': X_test_scaled,
            'y_test': y_test.values,
            'feature_names': self.feature_names,
            'class_weights': class_weights,
            'train_size': len(X_train_scaled),
            'val_size': len(X_val_scaled),
            'test_size': len(X_test_scaled),
            'bot_ratio_train': y_train.mean(),
            'bot_ratio_val': y_val.mean(),
            'bot_ratio_test': y_test.mean()
        }

    def save_preprocessor(self, filepath: str) -> None:
        """Save preprocessor to disk."""
        self.preprocessor.save(filepath)

    def load_preprocessor(self, filepath: str) -> None:
        """Load preprocessor from disk."""
        self.preprocessor.load(filepath)


if __name__ == '__main__':
    # Example usage
    from src.data.data_generator import DatasetGenerator
    from src.features.feature_extractor import FeatureEngineeringPipeline
    
    # Generate sample data
    generator = DatasetGenerator(seed=42)
    activity_df, network_df = generator.generate_dataset(num_humans=100, num_bots=50, days=30)
    
    # Extract features
    feature_pipeline = FeatureEngineeringPipeline()
    features_df = feature_pipeline.extract_all_features(activity_df, network_df)
    
    # Prepare data
    prep_pipeline = CompletePipeline(scaler_type='robust', handle_imbalance=True)
    data = prep_pipeline.prepare_data(
        features_df,
        target_column='is_bot',
        test_size=0.2,
        val_size=0.15,
        apply_smote=True
    )
    
    print("Data Preparation Complete!")
    print(f"Training samples: {data['train_size']} (bot ratio: {data['bot_ratio_train']:.2%})")
    print(f"Validation samples: {data['val_size']} (bot ratio: {data['bot_ratio_val']:.2%})")
    print(f"Test samples: {data['test_size']} (bot ratio: {data['bot_ratio_test']:.2%})")
    print(f"Features: {len(data['feature_names'])}")
    print(f"Class weights: {data['class_weights']}")

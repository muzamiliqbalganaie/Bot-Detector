"""
Model Calibration and Anomaly Detection
Platt Scaling for calibration and Isolation Forest for anomaly detection.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict
import joblib


class PlattScaler:
    """Platt Scaling for model calibration."""

    def __init__(self):
        """Initialize Platt scaler."""
        self.model = LogisticRegression(max_iter=1000)
        self.is_fitted = False

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        Fit Platt scaler on validation data.
        
        Args:
            y_pred: Predicted probabilities from model
            y_true: True labels
        """
        # Platt scaling: fit logistic regression on model outputs
        self.model.fit(y_pred.reshape(-1, 1), y_true)
        self.is_fitted = True

    def calibrate(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Calibrate predictions using fitted Platt scaler.
        
        Args:
            y_pred: Raw predicted probabilities
        
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        return self.model.predict_proba(y_pred.reshape(-1, 1))[:, 1]

    def save(self, filepath: str) -> None:
        """Save calibrator to disk."""
        joblib.dump(self.model, filepath)

    def load(self, filepath: str) -> None:
        """Load calibrator from disk."""
        self.model = joblib.load(filepath)
        self.is_fitted = True


class IsotonicRegression:
    """Isotonic Regression for model calibration (alternative to Platt)."""

    def __init__(self):
        """Initialize isotonic regression."""
        from sklearn.isotonic import IsotonicRegression as IsoReg
        self.model = IsoReg(out_of_bounds='clip')
        self.is_fitted = False

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Fit isotonic regression."""
        self.model.fit(y_pred, y_true)
        self.is_fitted = True

    def calibrate(self, y_pred: np.ndarray) -> np.ndarray:
        """Calibrate predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(y_pred)

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        joblib.dump(self.model, filepath)

    def load(self, filepath: str) -> None:
        """Load model from disk."""
        self.model = joblib.load(filepath)
        self.is_fitted = True


class AnomalyDetector:
    """Isolation Forest for anomaly detection."""

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of isolation trees
            random_state: Random seed
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> None:
        """
        Fit anomaly detector on normal data.
        
        Args:
            X: Training features (should be mostly normal data)
        """
        self.model.fit(X)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            X: Input features
        
        Returns:
            -1 for anomalies, 1 for normal
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (higher = more anomalous).
        
        Args:
            X: Input features
        
        Returns:
            Anomaly scores in range [0, 1]
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get raw anomaly scores
        scores = self.model.score_samples(X)
        
        # Normalize to [0, 1] range
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            normalized_scores = np.zeros_like(scores)
        else:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        
        # Invert so higher = more anomalous
        return 1 - normalized_scores

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        joblib.dump(self.model, filepath)

    def load(self, filepath: str) -> None:
        """Load model from disk."""
        self.model = joblib.load(filepath)
        self.is_fitted = True


class ConfidenceIntervalCalculator:
    """Calculate confidence intervals for predictions."""

    @staticmethod
    def calculate_wilson_ci(
        successes: int,
        trials: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate Wilson score confidence interval.
        
        Args:
            successes: Number of successes
            trials: Total number of trials
            confidence: Confidence level (0-1)
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        from scipy import stats
        
        if trials == 0:
            return 0, 1
        
        p_hat = successes / trials
        z = stats.norm.ppf((1 + confidence) / 2)
        
        denominator = 1 + z**2 / trials
        center = (p_hat + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denominator
        
        return max(0, center - margin), min(1, center + margin)

    @staticmethod
    def calculate_bootstrap_ci(
        predictions: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate bootstrap confidence intervals.
        
        Args:
            predictions: Array of predictions
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
        
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bounds = np.percentile(predictions, lower_percentile)
        upper_bounds = np.percentile(predictions, upper_percentile)
        
        return lower_bounds, upper_bounds


class CalibratedPredictor:
    """Complete calibrated prediction system."""

    def __init__(self, calibration_method: str = 'platt'):
        """
        Initialize calibrated predictor.
        
        Args:
            calibration_method: 'platt' or 'isotonic'
        """
        if calibration_method == 'platt':
            self.calibrator = PlattScaler()
        else:
            self.calibrator = IsotonicRegression()
        
        self.anomaly_detector = AnomalyDetector(contamination=0.1)
        self.is_calibrated = False

    def fit_calibration(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Fit calibration on validation data."""
        self.calibrator.fit(y_pred, y_true)
        self.is_calibrated = True

    def fit_anomaly_detection(self, X_normal: np.ndarray) -> None:
        """Fit anomaly detector on normal data."""
        self.anomaly_detector.fit(X_normal)

    def predict_with_calibration(
        self,
        y_pred: np.ndarray,
        X: np.ndarray = None,
        return_confidence: bool = True
    ) -> Dict:
        """
        Get calibrated predictions with confidence intervals and anomaly scores.
        
        Args:
            y_pred: Raw model predictions
            X: Input features (for anomaly detection)
            return_confidence: Whether to return confidence intervals
        
        Returns:
            Dictionary with calibrated predictions and metadata
        """
        if not self.is_calibrated:
            raise ValueError("Calibrator not fitted. Call fit_calibration() first.")
        
        # Calibrate predictions
        calibrated_proba = self.calibrator.calibrate(y_pred)
        
        result = {
            'raw_probability': y_pred,
            'calibrated_probability': calibrated_proba,
            'is_bot': (calibrated_proba >= 0.5).astype(int),
            'confidence': np.abs(calibrated_proba - 0.5) * 2
        }
        
        # Add anomaly scores if features provided
        if X is not None:
            anomaly_scores = self.anomaly_detector.predict_proba(X)
            result['anomaly_score'] = anomaly_scores
            result['is_anomaly'] = (anomaly_scores > 0.7).astype(int)
        
        # Add confidence intervals
        if return_confidence:
            lower, upper = ConfidenceIntervalCalculator.calculate_wilson_ci(
                int(calibrated_proba.sum()),
                len(calibrated_proba),
                confidence=0.95
            )
            result['confidence_interval'] = (lower, upper)
        
        return result

    def save(self, calibrator_path: str, anomaly_path: str) -> None:
        """Save both models."""
        self.calibrator.save(calibrator_path)
        self.anomaly_detector.save(anomaly_path)

    def load(self, calibrator_path: str, anomaly_path: str) -> None:
        """Load both models."""
        self.calibrator.load(calibrator_path)
        self.anomaly_detector.load(anomaly_path)
        self.is_calibrated = True


if __name__ == '__main__':
    # Example usage
    print("Calibration and Anomaly Detection")
    print("=" * 50)
    
    # Create dummy data
    y_pred_val = np.random.uniform(0, 1, 100)
    y_true_val = np.random.randint(0, 2, 100)
    X_normal = np.random.randn(100, 20)
    X_test = np.random.randn(50, 20)
    y_pred_test = np.random.uniform(0, 1, 50)
    
    # Initialize calibrated predictor
    predictor = CalibratedPredictor(calibration_method='platt')
    
    # Fit calibration
    predictor.fit_calibration(y_pred_val, y_true_val)
    
    # Fit anomaly detection
    predictor.fit_anomaly_detection(X_normal)
    
    # Predict with calibration
    results = predictor.predict_with_calibration(y_pred_test, X_test)
    
    print(f"Raw probabilities (first 5): {results['raw_probability'][:5]}")
    print(f"Calibrated probabilities (first 5): {results['calibrated_probability'][:5]}")
    print(f"Anomaly scores (first 5): {results['anomaly_score'][:5]}")
    print(f"Confidence intervals: {results['confidence_interval']}")

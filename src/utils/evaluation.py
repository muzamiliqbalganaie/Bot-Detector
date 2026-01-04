"""
Model Evaluation and Monitoring
Comprehensive evaluation metrics, confusion matrices, and performance monitoring.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, auc
)
from typing import Dict, Tuple, Optional
import json
from datetime import datetime


class EvaluationMetrics:
    """Calculate comprehensive evaluation metrics."""

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'specificity': EvaluationMetrics._calculate_specificity(y_true, y_pred),
            'false_positive_rate': EvaluationMetrics._calculate_fpr(y_true, y_pred),
            'false_negative_rate': EvaluationMetrics._calculate_fnr(y_true, y_pred)
        }
        
        # Add probability-based metrics if available
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
            # Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_auc'] = auc(recall, precision)
        
        return metrics

    @staticmethod
    def _calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        if tn + fp == 0:
            return 0
        return tn / (tn + fp)

    @staticmethod
    def _calculate_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate false positive rate."""
        return 1 - EvaluationMetrics._calculate_specificity(y_true, y_pred)

    @staticmethod
    def _calculate_fnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate false negative rate."""
        return 1 - recall_score(y_true, y_pred, zero_division=0)


class ConfusionMatrixAnalyzer:
    """Analyze and visualize confusion matrices."""

    @staticmethod
    def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute confusion matrix and derived metrics.
        
        Returns:
            Dictionary with confusion matrix and metrics
        """
        cm = confusion_matrix(y_true, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
        }

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        filepath: str = 'confusion_matrix.png'
    ) -> None:
        """Plot confusion matrix."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Failed to plot confusion matrix: {e}")


class ROCAnalyzer:
    """Analyze ROC curves and AUC."""

    @staticmethod
    def compute_roc_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict:
        """
        Compute ROC curve.
        
        Returns:
            Dictionary with FPR, TPR, thresholds, and AUC
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(roc_auc)
        }

    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        filepath: str = 'roc_curve.png'
    ) -> None:
        """Plot ROC curve."""
        try:
            import matplotlib.pyplot as plt
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Failed to plot ROC curve: {e}")


class PrecisionRecallAnalyzer:
    """Analyze precision-recall curves."""

    @staticmethod
    def compute_pr_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict:
        """Compute precision-recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(pr_auc)
        }

    @staticmethod
    def plot_pr_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        filepath: str = 'pr_curve.png'
    ) -> None:
        """Plot precision-recall curve."""
        try:
            import matplotlib.pyplot as plt
            
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Failed to plot PR curve: {e}")


class PerformanceMonitor:
    """Monitor model performance over time."""

    def __init__(self):
        """Initialize performance monitor."""
        self.history = []

    def log_prediction_batch(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        batch_name: str = None
    ) -> None:
        """Log a batch of predictions."""
        metrics = EvaluationMetrics.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'batch_name': batch_name,
            'sample_count': len(y_true),
            'bot_count': int(y_true.sum()),
            'metrics': metrics
        }
        
        self.history.append(log_entry)

    def get_performance_summary(self) -> Dict:
        """Get summary of performance over time."""
        if not self.history:
            return {}
        
        # Calculate average metrics across all batches
        all_metrics = [entry['metrics'] for entry in self.history]
        
        summary = {}
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            summary[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return summary

    def save_history(self, filepath: str) -> None:
        """Save monitoring history to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_history(self, filepath: str) -> None:
        """Load monitoring history from JSON."""
        with open(filepath, 'r') as f:
            self.history = json.load(f)


class ModelDriftDetector:
    """Detect model performance drift over time."""

    @staticmethod
    def detect_drift(
        baseline_metrics: Dict,
        current_metrics: Dict,
        threshold: float = 0.05
    ) -> Dict:
        """
        Detect if model performance has drifted.
        
        Args:
            baseline_metrics: Baseline metrics dictionary
            current_metrics: Current metrics dictionary
            threshold: Drift threshold (default 5%)
        
        Returns:
            Dictionary with drift detection results
        """
        drift_report = {
            'has_drift': False,
            'drifted_metrics': [],
            'drift_details': {}
        }
        
        for metric_name in baseline_metrics.keys():
            if metric_name not in current_metrics:
                continue
            
            baseline_val = baseline_metrics[metric_name]
            current_val = current_metrics[metric_name]
            
            # Calculate percentage change
            if baseline_val != 0:
                pct_change = abs(current_val - baseline_val) / abs(baseline_val)
            else:
                pct_change = abs(current_val - baseline_val)
            
            if pct_change > threshold:
                drift_report['has_drift'] = True
                drift_report['drifted_metrics'].append(metric_name)
                drift_report['drift_details'][metric_name] = {
                    'baseline': baseline_val,
                    'current': current_val,
                    'pct_change': pct_change
                }
        
        return drift_report


class ComprehensiveEvaluator:
    """Complete evaluation pipeline."""

    def __init__(self):
        """Initialize evaluator."""
        self.metrics_calc = EvaluationMetrics()
        self.cm_analyzer = ConfusionMatrixAnalyzer()
        self.roc_analyzer = ROCAnalyzer()
        self.pr_analyzer = PrecisionRecallAnalyzer()
        self.monitor = PerformanceMonitor()

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        output_dir: str = 'evaluation_results'
    ) -> Dict:
        """
        Complete model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            output_dir: Directory to save results
        
        Returns:
            Comprehensive evaluation report
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate metrics
        metrics = self.metrics_calc.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Confusion matrix
        cm_analysis = self.cm_analyzer.compute_confusion_matrix(y_true, y_pred)
        
        # ROC curve
        roc_analysis = self.roc_analyzer.compute_roc_curve(y_true, y_pred_proba)
        
        # PR curve
        pr_analysis = self.pr_analyzer.compute_pr_curve(y_true, y_pred_proba)
        
        # Generate visualizations
        self.cm_analyzer.plot_confusion_matrix(y_true, y_pred, f'{output_dir}/confusion_matrix.png')
        self.roc_analyzer.plot_roc_curve(y_true, y_pred_proba, f'{output_dir}/roc_curve.png')
        self.pr_analyzer.plot_pr_curve(y_true, y_pred_proba, f'{output_dir}/pr_curve.png')
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'confusion_matrix': cm_analysis,
            'roc_curve': roc_analysis,
            'pr_curve': pr_analysis,
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        # Save report
        with open(f'{output_dir}/evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


if __name__ == '__main__':
    # Example usage
    print("Model Evaluation")
    print("=" * 50)
    
    # Create dummy data
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_pred_proba = np.random.uniform(0, 1, 100)
    
    # Evaluate
    evaluator = ComprehensiveEvaluator()
    report = evaluator.evaluate_model(y_true, y_pred, y_pred_proba)
    
    print("Evaluation Report:")
    print(f"Accuracy: {report['metrics']['accuracy']:.4f}")
    print(f"Precision: {report['metrics']['precision']:.4f}")
    print(f"Recall: {report['metrics']['recall']:.4f}")
    print(f"F1 Score: {report['metrics']['f1']:.4f}")
    print(f"ROC AUC: {report['metrics'].get('roc_auc', 'N/A')}")

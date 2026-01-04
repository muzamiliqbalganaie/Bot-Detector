"""Bot Detection System Package"""

__version__ = "1.0.0"
__author__ = "Bot Detection Team"

from src.data.data_generator import DatasetGenerator, UserActivityGenerator
from src.features.feature_extractor import FeatureExtractor, FeatureEngineeringPipeline
from src.features.preprocessing import DataPreprocessor, CompletePipeline
from src.models.two_stage_model import TwoStageDetector, FastFilterModel, GRUSequenceModel
from src.models.calibration_anomaly import CalibratedPredictor, PlattScaler, AnomalyDetector
from src.utils.evaluation import ComprehensiveEvaluator, EvaluationMetrics

__all__ = [
    'DatasetGenerator',
    'UserActivityGenerator',
    'FeatureExtractor',
    'FeatureEngineeringPipeline',
    'DataPreprocessor',
    'CompletePipeline',
    'TwoStageDetector',
    'FastFilterModel',
    'GRUSequenceModel',
    'CalibratedPredictor',
    'PlattScaler',
    'AnomalyDetector',
    'ComprehensiveEvaluator',
    'EvaluationMetrics'
]

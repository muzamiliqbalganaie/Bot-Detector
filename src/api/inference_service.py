"""
FastAPI Inference Service
Production-ready API for bot detection predictions.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import logging
import json
from datetime import datetime
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserActivityData(BaseModel):
    """User activity data for prediction."""
    user_id: int
    posting_cadence: float = Field(..., description="Average time between posts (seconds)")
    posting_cadence_std: float = Field(..., description="Std dev of inter-post times")
    burst_score: float = Field(..., description="Indicator of activity spikes")
    circadian_entropy: float = Field(..., description="Activity distribution across hours")
    active_hours: int = Field(..., description="Number of active hours")
    action_diversity: float = Field(..., description="Entropy of action types")
    typo_rate: float = Field(..., description="Fraction of posts with typos")
    session_duration_mean: float = Field(..., description="Average session length")
    session_duration_std: float = Field(..., description="Std dev of session lengths")
    device_diversity: int = Field(..., description="Number of unique devices")
    content_length_mean: float = Field(..., description="Average content length")
    content_length_std: float = Field(..., description="Std dev of content length")
    content_length_variance: float = Field(..., description="Variance in post lengths")
    duplicate_rate: float = Field(..., description="Rate of duplicate content")
    content_diversity: float = Field(..., description="Semantic diversity")
    follower_following_ratio: float = Field(..., description="Follower/following ratio")
    account_age_days: int = Field(..., description="Account age in days")
    is_verified: int = Field(..., description="Whether account is verified")
    profile_completeness: float = Field(..., description="Profile completeness score")
    followers: int = Field(..., description="Number of followers")
    following: int = Field(..., description="Number of following")


class PredictionRequest(BaseModel):
    """Request for batch predictions."""
    users: List[UserActivityData]
    return_confidence_interval: bool = True
    return_anomaly_score: bool = True


class PredictionResponse(BaseModel):
    """Single prediction response."""
    user_id: int
    bot_probability: float
    is_bot: int
    confidence: float
    anomaly_score: Optional[float] = None
    is_anomaly: Optional[int] = None
    confidence_interval: Optional[tuple] = None
    prediction_timestamp: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total_predictions: int
    bot_count: int
    human_count: int
    average_bot_probability: float
    processing_time_ms: float


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str
    version: str = "1.0.0"


class ModelManager:
    """Manages model loading and predictions."""

    def __init__(self, model_dir: str = 'models'):
        """Initialize model manager."""
        self.model_dir = model_dir
        self.fast_filter = None
        self.sequence_trainer = None
        self.calibrator = None
        self.preprocessor = None
        self.is_loaded = False
        self.feature_names = None

    def load_models(self) -> bool:
        """Load all models from disk."""
        try:
            import joblib
            from src.models.two_stage_model import FastFilterModel, SequenceModelTrainer, GRUSequenceModel
            from src.models.calibration_anomaly import CalibratedPredictor
            from src.features.preprocessing import DataPreprocessor
            
            # Load fast filter
            filter_path = os.path.join(self.model_dir, 'fast_filter.pkl')
            if os.path.exists(filter_path):
                self.fast_filter = FastFilterModel()
                self.fast_filter.load(filter_path)
                logger.info("Loaded fast filter model")
            
            # Load preprocessor
            preprocessor_path = os.path.join(self.model_dir, 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                self.preprocessor = DataPreprocessor()
                self.preprocessor.load(preprocessor_path)
                self.feature_names = self.preprocessor.feature_names
                logger.info("Loaded preprocessor")
            
            # Load calibrator
            calibrator_path = os.path.join(self.model_dir, 'calibrator.pkl')
            if os.path.exists(calibrator_path):
                self.calibrator = CalibratedPredictor()
                self.calibrator.load(
                    calibrator_path,
                    os.path.join(self.model_dir, 'anomaly_detector.pkl')
                )
                logger.info("Loaded calibrator and anomaly detector")
            
            self.is_loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def predict(self, features: np.ndarray) -> Dict:
        """Make prediction on features."""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded")
        
        # Get raw predictions from fast filter
        raw_proba = self.fast_filter.predict(features)
        
        # Calibrate predictions
        if self.calibrator:
            result = self.calibrator.predict_with_calibration(
                raw_proba,
                X=features,
                return_confidence=True
            )
        else:
            result = {
                'calibrated_probability': raw_proba,
                'is_bot': (raw_proba >= 0.5).astype(int),
                'confidence': np.abs(raw_proba - 0.5) * 2
            }
        
        return result


def create_app(model_dir: str = 'models') -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Bot Detection API",
        description="Production-ready bot detection inference service",
        version="1.0.0"
    )
    
    # Initialize model manager
    model_manager = ModelManager(model_dir=model_dir)
    
    @app.on_event("startup")
    async def startup_event():
        """Load models on startup."""
        logger.info("Starting up Bot Detection API")
        if model_manager.load_models():
            logger.info("Models loaded successfully")
        else:
            logger.warning("Failed to load models - running in demo mode")

    @app.get("/health", response_model=HealthCheckResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthCheckResponse(
            status="healthy",
            model_loaded=model_manager.is_loaded,
            timestamp=datetime.now().isoformat()
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict_single(user_data: UserActivityData):
        """Predict bot probability for a single user."""
        try:
            # Convert user data to feature array
            features = np.array([[
                user_data.posting_cadence,
                user_data.posting_cadence_std,
                user_data.burst_score,
                user_data.circadian_entropy,
                user_data.active_hours,
                user_data.action_diversity,
                user_data.typo_rate,
                user_data.session_duration_mean,
                user_data.session_duration_std,
                user_data.device_diversity,
                user_data.content_length_mean,
                user_data.content_length_std,
                user_data.content_length_variance,
                user_data.duplicate_rate,
                user_data.content_diversity,
                user_data.follower_following_ratio,
                user_data.account_age_days,
                user_data.is_verified,
                user_data.profile_completeness,
                user_data.followers,
                user_data.following
            ]])
            
            # Make prediction
            if model_manager.is_loaded:
                result = model_manager.predict(features)
                bot_prob = result['calibrated_probability'][0]
                is_bot = result['is_bot'][0]
                confidence = result['confidence'][0]
                anomaly_score = result.get('anomaly_score', [0])[0]
                is_anomaly = result.get('is_anomaly', [0])[0]
                ci = result.get('confidence_interval', (0, 1))
            else:
                # Demo mode
                bot_prob = np.random.uniform(0, 1)
                is_bot = int(bot_prob > 0.5)
                confidence = np.abs(bot_prob - 0.5) * 2
                anomaly_score = np.random.uniform(0, 1)
                is_anomaly = int(anomaly_score > 0.7)
                ci = (max(0, bot_prob - 0.1), min(1, bot_prob + 0.1))
            
            return PredictionResponse(
                user_id=user_data.user_id,
                bot_probability=float(bot_prob),
                is_bot=int(is_bot),
                confidence=float(confidence),
                anomaly_score=float(anomaly_score),
                is_anomaly=int(is_anomaly),
                confidence_interval=ci,
                prediction_timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict_batch", response_model=BatchPredictionResponse)
    async def predict_batch(request: PredictionRequest):
        """Predict bot probability for multiple users."""
        try:
            import time
            start_time = time.time()
            
            predictions = []
            for user_data in request.users:
                prediction = await predict_single(user_data)
                predictions.append(prediction)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            bot_count = sum(1 for p in predictions if p.is_bot == 1)
            human_count = len(predictions) - bot_count
            avg_bot_prob = np.mean([p.bot_probability for p in predictions])
            
            return BatchPredictionResponse(
                predictions=predictions,
                total_predictions=len(predictions),
                bot_count=bot_count,
                human_count=human_count,
                average_bot_probability=float(avg_bot_prob),
                processing_time_ms=processing_time_ms
            )
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/stats")
    async def get_stats():
        """Get API statistics."""
        return {
            "api_version": "1.0.0",
            "model_loaded": model_manager.is_loaded,
            "feature_count": len(model_manager.feature_names) if model_manager.feature_names else 0,
            "timestamp": datetime.now().isoformat()
        }

    return app


if __name__ == '__main__':
    import uvicorn
    
    app = create_app(model_dir='models')
    
    # Run the server
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8000,
        log_level='info'
    )

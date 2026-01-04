#!/usr/bin/env python3
"""
Complete Training Pipeline
Orchestrates data generation, feature engineering, model training, and evaluation.
"""

import os
import sys
import numpy as np
import argparse
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """Run complete training pipeline."""
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f'{args.output_dir}/models', exist_ok=True)
    os.makedirs(f'{args.output_dir}/data', exist_ok=True)
    os.makedirs(f'{args.output_dir}/evaluation', exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Bot Detection System - Complete Training Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Data Generation
    logger.info("\n[Step 1] Generating synthetic data...")
    from src.data.data_generator import DatasetGenerator
    
    generator = DatasetGenerator(seed=args.seed)
    activity_df, network_df = generator.generate_dataset(
        num_humans=args.num_humans,
        num_bots=args.num_bots,
        days=args.days,
        bot_type_distribution={
            'aggressive': 0.5,
            'moderate': 0.3,
            'sophisticated': 0.2
        }
    )
    
    generator.save_dataset(activity_df, network_df, f'{args.output_dir}/data')
    logger.info(f"Generated {len(activity_df)} activity records")
    logger.info(f"Bot ratio: {activity_df['is_bot'].sum() / len(activity_df):.2%}")
    
    # Step 2: Feature Engineering
    logger.info("\n[Step 2] Extracting features...")
    from src.features.feature_extractor import FeatureEngineeringPipeline
    
    feature_pipeline = FeatureEngineeringPipeline()
    features_df = feature_pipeline.extract_all_features(activity_df, network_df)
    
    features_df.to_csv(f'{args.output_dir}/data/features.csv', index=False)
    logger.info(f"Extracted {len(features_df)} user profiles with {len(feature_pipeline.get_feature_names())} features")
    
    # Step 3: Data Preprocessing
    logger.info("\n[Step 3] Preprocessing data...")
    from src.features.preprocessing import CompletePipeline
    
    prep_pipeline = CompletePipeline(scaler_type='robust', handle_imbalance=True)
    data = prep_pipeline.prepare_data(
        features_df,
        target_column='is_bot',
        test_size=args.test_size,
        val_size=args.val_size,
        apply_smote=True,
        random_state=args.seed
    )
    
    logger.info(f"Training samples: {data['train_size']} (bot ratio: {data['bot_ratio_train']:.2%})")
    logger.info(f"Validation samples: {data['val_size']} (bot ratio: {data['bot_ratio_val']:.2%})")
    logger.info(f"Test samples: {data['test_size']} (bot ratio: {data['bot_ratio_test']:.2%})")
    logger.info(f"Features after selection: {len(data['feature_names'])}")
    
    # Save preprocessor
    prep_pipeline.save_preprocessor(f'{args.output_dir}/models/preprocessor.pkl')
    
    # Step 4: Train Two-Stage Model
    logger.info("\n[Step 4] Training two-stage model...")
    from src.models.two_stage_model import TwoStageDetector
    
    detector = TwoStageDetector(device=args.device)
    
    # Train Stage 1
    stage1_metrics = detector.train_stage1(data['X_train'], data['y_train'])
    logger.info(f"Stage 1 - Accuracy: {stage1_metrics['accuracy']:.4f}, F1: {stage1_metrics['f1']:.4f}")
    
    # Train Stage 2
    history = detector.train_stage2(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        input_size=data['X_train'].shape[1],
        epochs=args.epochs
    )
    
    # Save models
    detector.save(
        f'{args.output_dir}/models/fast_filter.pkl',
        f'{args.output_dir}/models/sequence_model.pt'
    )
    logger.info("Models saved")
    
    # Step 5: Model Calibration
    logger.info("\n[Step 5] Calibrating model...")
    from src.models.calibration_anomaly import CalibratedPredictor
    
    # Get predictions on validation set
    val_predictions = detector.fast_filter.predict(data['X_val'])
    
    # Fit calibrator
    calibrator = CalibratedPredictor(calibration_method='platt')
    calibrator.fit_calibration(val_predictions, data['y_val'])
    
    # Fit anomaly detector on normal data
    normal_data = data['X_train'][data['y_train'] == 0]
    calibrator.fit_anomaly_detection(normal_data)
    
    # Save calibrator
    calibrator.save(
        f'{args.output_dir}/models/calibrator.pkl',
        f'{args.output_dir}/models/anomaly_detector.pkl'
    )
    logger.info("Calibration complete")
    
    # Step 6: Evaluation
    logger.info("\n[Step 6] Evaluating model...")
    from src.utils.evaluation import ComprehensiveEvaluator
    
    # Get test predictions
    test_predictions = detector.predict(data['X_test'], use_both_stages=True)
    test_pred_classes = test_predictions['is_bot']
    test_pred_proba = test_predictions['bot_probability']
    
    # Evaluate
    evaluator = ComprehensiveEvaluator()
    report = evaluator.evaluate_model(
        data['y_test'],
        test_pred_classes,
        test_pred_proba,
        output_dir=f'{args.output_dir}/evaluation'
    )
    
    logger.info(f"Test Accuracy: {report['metrics']['accuracy']:.4f}")
    logger.info(f"Test Precision: {report['metrics']['precision']:.4f}")
    logger.info(f"Test Recall: {report['metrics']['recall']:.4f}")
    logger.info(f"Test F1 Score: {report['metrics']['f1']:.4f}")
    logger.info(f"Test ROC AUC: {report['metrics'].get('roc_auc', 'N/A')}")
    
    # Step 7: Generate Report
    logger.info("\n[Step 7] Generating final report...")
    
    final_report = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'num_humans': args.num_humans,
            'num_bots': args.num_bots,
            'days': args.days,
            'test_size': args.test_size,
            'val_size': args.val_size,
            'epochs': args.epochs,
            'seed': args.seed
        },
        'data_summary': {
            'total_users': len(features_df),
            'total_activities': len(activity_df),
            'features': len(data['feature_names']),
            'train_size': data['train_size'],
            'val_size': data['val_size'],
            'test_size': data['test_size']
        },
        'performance_metrics': report['metrics'],
        'confusion_matrix': report['confusion_matrix'],
        'class_weights': {str(k): v for k, v in data['class_weights'].items()}
    }
    
    import json
    with open(f'{args.output_dir}/training_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bot Detection System - Training Pipeline')
    
    # Data generation arguments
    parser.add_argument('--num-humans', type=int, default=200, help='Number of human users')
    parser.add_argument('--num-bots', type=int, default=100, help='Number of bot users')
    parser.add_argument('--days', type=int, default=30, help='Days of activity to simulate')
    
    # Data split arguments
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--val-size', type=float, default=0.15, help='Validation set fraction')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='training_output', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

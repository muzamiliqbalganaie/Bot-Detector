"""
Feature Engineering Module
Extracts behavioral, temporal, content, and network features from user activity data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from scipy import stats


class FeatureExtractor:
    """Extracts features from user activity data."""

    @staticmethod
    def extract_temporal_features(activities: pd.DataFrame, user_id: int) -> Dict:
        """
        Extract temporal features from user activity.
        
        Features:
        - posting_cadence: Average time between posts (seconds)
        - posting_cadence_std: Standard deviation of inter-post times
        - burst_score: Indicator of sudden activity spikes
        - circadian_entropy: Measure of activity distribution across hours
        """
        user_activities = activities[activities['user_id'] == user_id].sort_values('timestamp')
        
        if len(user_activities) < 2:
            return {
                'posting_cadence': 0,
                'posting_cadence_std': 0,
                'burst_score': 0,
                'circadian_entropy': 0,
                'active_hours': 0
            }
        
        # Calculate inter-event times
        timestamps = pd.to_datetime(user_activities['timestamp'])
        inter_event_times = timestamps.diff().dt.total_seconds().dropna()
        
        features = {
            'posting_cadence': inter_event_times.mean() if len(inter_event_times) > 0 else 0,
            'posting_cadence_std': inter_event_times.std() if len(inter_event_times) > 1 else 0,
            'burst_score': FeatureExtractor._calculate_burst_score(inter_event_times),
            'circadian_entropy': FeatureExtractor._calculate_circadian_entropy(timestamps),
            'active_hours': len(set(timestamps.dt.hour))
        }
        
        return features

    @staticmethod
    def extract_behavioral_features(activities: pd.DataFrame, user_id: int) -> Dict:
        """
        Extract behavioral features from user activity.
        
        Features:
        - action_diversity: Entropy of action types
        - typo_rate: Fraction of posts with typos
        - session_duration_mean: Average session length
        - session_duration_std: Std dev of session lengths
        - device_diversity: Number of unique devices used
        """
        user_activities = activities[activities['user_id'] == user_id]
        
        if len(user_activities) == 0:
            return {
                'action_diversity': 0,
                'typo_rate': 0,
                'session_duration_mean': 0,
                'session_duration_std': 0,
                'device_diversity': 0,
                'content_length_mean': 0,
                'content_length_std': 0
            }
        
        # Action diversity (entropy)
        action_counts = user_activities['action_type'].value_counts()
        action_probs = action_counts / len(user_activities)
        action_entropy = -np.sum(action_probs * np.log2(action_probs + 1e-10))
        
        # Typo rate
        typo_rate = user_activities['has_typo'].sum() / len(user_activities)
        
        # Session duration statistics
        session_durations = user_activities['session_duration_seconds']
        
        features = {
            'action_diversity': action_entropy,
            'typo_rate': typo_rate,
            'session_duration_mean': session_durations.mean(),
            'session_duration_std': session_durations.std(),
            'device_diversity': user_activities['device_type'].nunique(),
            'content_length_mean': user_activities['content_length'].mean(),
            'content_length_std': user_activities['content_length'].std()
        }
        
        return features

    @staticmethod
    def extract_content_features(activities: pd.DataFrame, user_id: int) -> Dict:
        """
        Extract content-based features.
        
        Features:
        - content_length_variance: Variance in post lengths
        - duplicate_rate: Estimated rate of duplicate content
        - content_diversity: Semantic diversity (simplified)
        """
        user_activities = activities[activities['user_id'] == user_id]
        
        if len(user_activities) == 0:
            return {
                'content_length_variance': 0,
                'duplicate_rate': 0,
                'content_diversity': 0
            }
        
        content_lengths = user_activities['content_length']
        
        # Content length variance
        content_variance = content_lengths.var()
        
        # Duplicate rate (simplified: based on content length similarity)
        # In production, use embedding similarity
        length_counts = content_lengths.value_counts()
        duplicate_rate = (length_counts[length_counts > 1].sum() / len(user_activities)) if len(length_counts) > 0 else 0
        
        # Content diversity (entropy of content lengths)
        content_bins = pd.cut(content_lengths, bins=10)
        content_dist = content_bins.value_counts()
        content_probs = content_dist / len(user_activities)
        content_diversity = -np.sum(content_probs * np.log2(content_probs + 1e-10))
        
        features = {
            'content_length_variance': content_variance,
            'duplicate_rate': duplicate_rate,
            'content_diversity': content_diversity
        }
        
        return features

    @staticmethod
    def extract_network_features(network_df: pd.DataFrame, user_id: int) -> Dict:
        """Extract network topology features."""
        user_network = network_df[network_df['user_id'] == user_id]
        
        if len(user_network) == 0:
            return {
                'follower_following_ratio': 0,
                'account_age_days': 0,
                'is_verified': 0,
                'profile_completeness': 0,
                'followers': 0,
                'following': 0
            }
        
        row = user_network.iloc[0]
        
        features = {
            'follower_following_ratio': row['follower_following_ratio'],
            'account_age_days': row['account_age_days'],
            'is_verified': int(row['is_verified']),
            'profile_completeness': row['profile_completeness'],
            'followers': row['followers'],
            'following': row['following']
        }
        
        return features

    @staticmethod
    def _calculate_burst_score(inter_event_times: pd.Series) -> float:
        """Calculate burst score indicating sudden activity spikes."""
        if len(inter_event_times) < 2:
            return 0
        
        # Burst score: ratio of very short intervals to total intervals
        very_short_intervals = (inter_event_times < inter_event_times.quantile(0.25)).sum()
        burst_score = very_short_intervals / len(inter_event_times)
        
        return burst_score

    @staticmethod
    def _calculate_circadian_entropy(timestamps: pd.Series) -> float:
        """Calculate entropy of activity distribution across hours."""
        hour_counts = timestamps.dt.hour.value_counts()
        hour_probs = hour_counts / len(timestamps)
        entropy = -np.sum(hour_probs * np.log2(hour_probs + 1e-10))
        
        # Normalize by maximum entropy (log2(24))
        max_entropy = np.log2(24)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy


class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline."""

    def __init__(self):
        """Initialize the pipeline."""
        self.feature_extractor = FeatureExtractor()
        self.scaler = None
        self.feature_names = None

    def extract_all_features(
        self,
        activities: pd.DataFrame,
        network_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract all features for all users.
        
        Returns:
            DataFrame with one row per user and all features as columns
        """
        all_users = pd.concat([
            activities['user_id'],
            network_df['user_id']
        ]).unique()
        
        features_list = []
        
        for user_id in all_users:
            user_features = {'user_id': user_id}
            
            # Extract all feature groups
            temporal_features = self.feature_extractor.extract_temporal_features(activities, user_id)
            behavioral_features = self.feature_extractor.extract_behavioral_features(activities, user_id)
            content_features = self.feature_extractor.extract_content_features(activities, user_id)
            network_features = self.feature_extractor.extract_network_features(network_df, user_id)
            
            # Get ground truth label
            user_activities = activities[activities['user_id'] == user_id]
            is_bot = user_activities['is_bot'].iloc[0] if len(user_activities) > 0 else False
            
            # Combine all features
            user_features.update(temporal_features)
            user_features.update(behavioral_features)
            user_features.update(content_features)
            user_features.update(network_features)
            user_features['is_bot'] = int(is_bot)
            
            features_list.append(user_features)
        
        features_df = pd.DataFrame(features_list)
        self.feature_names = [col for col in features_df.columns if col not in ['user_id', 'is_bot']]
        
        return features_df

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names if self.feature_names else []


if __name__ == '__main__':
    # Example usage
    from src.data.data_generator import DatasetGenerator
    
    # Generate sample data
    generator = DatasetGenerator(seed=42)
    activity_df, network_df = generator.generate_dataset(num_humans=50, num_bots=25, days=30)
    
    # Extract features
    pipeline = FeatureEngineeringPipeline()
    features_df = pipeline.extract_all_features(activity_df, network_df)
    
    print("Extracted Features:")
    print(features_df.head())
    print(f"\nShape: {features_df.shape}")
    print(f"\nFeature names: {pipeline.get_feature_names()}")
    print(f"\nBot distribution:")
    print(features_df['is_bot'].value_counts())

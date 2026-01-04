"""
Data Simulation Pipeline
Generates realistic synthetic user activity patterns with configurable bot/human ratios.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import json


class UserActivityGenerator:
    """Generates synthetic user activity data with realistic patterns."""

    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed for reproducibility."""
        np.random.seed(seed)
        self.seed = seed

    def generate_human_activity(self, user_id: int, days: int = 30) -> List[Dict]:
        """
        Generate realistic human activity patterns.
        
        Humans exhibit:
        - Circadian rhythms (more active during day)
        - Variable session durations
        - Natural inter-event time distributions
        - Typos and corrections
        """
        activities = []
        base_date = datetime.now() - timedelta(days=days)
        
        # Human posting frequency: 1-5 posts per day
        daily_posts = np.random.randint(1, 6)
        
        for day in range(days):
            current_date = base_date + timedelta(days=day)
            
            # Circadian rhythm: more active 8am-11pm
            hour_bias = np.random.choice(
                range(24),
                size=daily_posts,
                p=self._circadian_distribution()
            )
            
            for hour in hour_bias:
                timestamp = current_date.replace(
                    hour=hour,
                    minute=np.random.randint(0, 60),
                    second=np.random.randint(0, 60)
                )
                
                activities.append({
                    'user_id': user_id,
                    'timestamp': timestamp.isoformat(),
                    'action_type': np.random.choice(['post', 'like', 'comment', 'share']),
                    'content_length': np.random.randint(10, 500),
                    'has_typo': np.random.choice([True, False], p=[0.05, 0.95]),
                    'session_duration_seconds': np.random.exponential(300),  # avg 5 min
                    'device_type': np.random.choice(['mobile', 'desktop', 'tablet']),
                    'is_bot': False
                })
        
        return activities

    def generate_bot_activity(self, user_id: int, days: int = 30, bot_type: str = 'aggressive') -> List[Dict]:
        """
        Generate bot activity patterns.
        
        Bots exhibit:
        - Uniform distribution across hours (no circadian rhythm)
        - Consistent session patterns
        - High posting frequency
        - Duplicate or near-duplicate content
        - Precise timing (no human variability)
        """
        activities = []
        base_date = datetime.now() - timedelta(days=days)
        
        if bot_type == 'aggressive':
            # Aggressive bots: 20-50 posts per day
            daily_posts = np.random.randint(20, 51)
            inter_event_seconds = np.random.randint(60, 300)  # 1-5 min intervals
        elif bot_type == 'moderate':
            # Moderate bots: 5-15 posts per day
            daily_posts = np.random.randint(5, 16)
            inter_event_seconds = np.random.randint(300, 1800)  # 5-30 min intervals
        else:  # sophisticated
            # Sophisticated bots: 2-8 posts per day, mimics human patterns
            daily_posts = np.random.randint(2, 9)
            inter_event_seconds = np.random.randint(1800, 7200)  # 30 min - 2 hours
        
        for day in range(days):
            current_date = base_date + timedelta(days=day)
            
            # Uniform distribution (no circadian rhythm)
            hours = np.random.choice(range(24), size=daily_posts, replace=True)
            
            for hour in hours:
                timestamp = current_date.replace(
                    hour=hour,
                    minute=np.random.randint(0, 60),
                    second=np.random.randint(0, 60)
                )
                
                activities.append({
                    'user_id': user_id,
                    'timestamp': timestamp.isoformat(),
                    'action_type': np.random.choice(['post', 'like', 'comment', 'share']),
                    'content_length': np.random.randint(50, 200) if bot_type == 'aggressive' else np.random.randint(10, 500),
                    'has_typo': False,  # Bots don't make typos
                    'session_duration_seconds': 5.0 if bot_type == 'aggressive' else np.random.exponential(300),
                    'device_type': 'unknown',  # Bots often have consistent device
                    'is_bot': True
                })
        
        return activities

    def generate_network_features(self, user_id: int, is_bot: bool) -> Dict:
        """Generate network topology features."""
        if is_bot:
            # Bots have skewed follower/following ratios
            followers = np.random.randint(100, 1000)
            following = np.random.randint(5000, 50000)
        else:
            # Humans have more balanced ratios
            followers = np.random.randint(50, 5000)
            following = np.random.randint(100, 2000)
        
        return {
            'user_id': user_id,
            'followers': followers,
            'following': following,
            'follower_following_ratio': followers / (following + 1),
            'account_age_days': np.random.randint(1, 1000),
            'is_verified': np.random.choice([True, False], p=[0.1, 0.9]),
            'profile_completeness': np.random.uniform(0, 1)
        }

    def _circadian_distribution(self) -> np.ndarray:
        """Return probability distribution for human circadian rhythm."""
        # More active during day (8am-11pm), less active at night
        distribution = np.array([
            0.01, 0.01, 0.01, 0.01, 0.02, 0.03,  # 0-5am: very low
            0.05, 0.08, 0.12, 0.12, 0.10, 0.10,  # 6-11am: increasing
            0.10, 0.10, 0.10, 0.10, 0.10, 0.10,  # 12-5pm: high
            0.09, 0.08, 0.07, 0.06, 0.04, 0.02   # 6-11pm: decreasing
        ])
        return distribution / distribution.sum()


class DatasetGenerator:
    """Generates complete datasets with mixed human and bot users."""

    def __init__(self, seed: int = 42):
        """Initialize dataset generator."""
        self.activity_gen = UserActivityGenerator(seed=seed)
        self.seed = seed

    def generate_dataset(
        self,
        num_humans: int = 100,
        num_bots: int = 50,
        days: int = 30,
        bot_type_distribution: Dict[str, float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate a complete dataset with activity and network features.
        
        Args:
            num_humans: Number of human users
            num_bots: Number of bot users
            days: Number of days of activity to simulate
            bot_type_distribution: Distribution of bot types (aggressive, moderate, sophisticated)
        
        Returns:
            Tuple of (activity_df, network_df)
        """
        if bot_type_distribution is None:
            bot_type_distribution = {'aggressive': 0.5, 'moderate': 0.3, 'sophisticated': 0.2}
        
        all_activities = []
        all_network_features = []
        
        # Generate human users
        for user_id in range(num_humans):
            activities = self.activity_gen.generate_human_activity(user_id, days=days)
            all_activities.extend(activities)
            
            network_features = self.activity_gen.generate_network_features(user_id, is_bot=False)
            all_network_features.append(network_features)
        
        # Generate bot users
        for user_id in range(num_humans, num_humans + num_bots):
            bot_type = np.random.choice(
                list(bot_type_distribution.keys()),
                p=list(bot_type_distribution.values())
            )
            activities = self.activity_gen.generate_bot_activity(user_id, days=days, bot_type=bot_type)
            all_activities.extend(activities)
            
            network_features = self.activity_gen.generate_network_features(user_id, is_bot=True)
            all_network_features.append(network_features)
        
        # Convert to DataFrames
        activity_df = pd.DataFrame(all_activities)
        network_df = pd.DataFrame(all_network_features)
        
        # Convert timestamp to datetime
        activity_df['timestamp'] = pd.to_datetime(activity_df['timestamp'])
        
        return activity_df, network_df

    def save_dataset(self, activity_df: pd.DataFrame, network_df: pd.DataFrame, output_dir: str = 'data'):
        """Save generated dataset to CSV files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        activity_df.to_csv(f'{output_dir}/activities.csv', index=False)
        network_df.to_csv(f'{output_dir}/network_features.csv', index=False)
        
        print(f"Dataset saved to {output_dir}/")
        print(f"Activities: {len(activity_df)} records")
        print(f"Network features: {len(network_df)} records")


if __name__ == '__main__':
    # Example usage
    generator = DatasetGenerator(seed=42)
    activity_df, network_df = generator.generate_dataset(
        num_humans=100,
        num_bots=50,
        days=30
    )
    
    print("Activity DataFrame:")
    print(activity_df.head())
    print(f"\nShape: {activity_df.shape}")
    print(f"Bot ratio: {activity_df['is_bot'].sum() / len(activity_df):.2%}")
    
    print("\nNetwork Features DataFrame:")
    print(network_df.head())
    print(f"Shape: {network_df.shape}")
    
    # Save dataset
    generator.save_dataset(activity_df, network_df, output_dir='data')

"""
External features preprocessing for DeepAR.

Handles multiple external features with flexible preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class ExternalFeaturesPreprocessor:
    """
    Preprocessor for external features (revenue, price, promotions, etc.).
    
    Supports:
    - Multiple features (dynamic_real and static_cat)
    - Lag creation
    - Rolling averages
    - Normalization
    - Missing value handling
    - Outlier clipping
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor.
        
        Args:
            config: External features configuration
        """
        self.config = config
        self.preprocessing_config = config.get('preprocessing', {})
        self.features_config = config.get('features', [])
        
        # Get enabled features
        self.enabled_features = [
            f for f in self.features_config
            if f.get('enabled', True)
        ]
        
        # Separate dynamic and static features
        self.dynamic_features = [
            f for f in self.enabled_features
            if f.get('type') == 'dynamic_real'
        ]
        
        self.static_features = [
            f for f in self.enabled_features
            if f.get('type') == 'static_cat'
        ]
        
        # Scalers for normalization
        self.scalers = {}
        
        logger.info(f"Initialized with {len(self.enabled_features)} features:")
        logger.info(f"  - Dynamic: {len(self.dynamic_features)}")
        logger.info(f"  - Static: {len(self.static_features)}")
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Get names of enabled features.
        
        Returns:
            Dictionary with 'dynamic' and 'static' feature lists
        """
        return {
            'dynamic': [f['name'] for f in self.dynamic_features],
            'static': [f['name'] for f in self.static_features]
        }
    
    def validate_features(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that required features exist in dataframe.
        
        Args:
            df: Input dataframe
        
        Returns:
            Tuple of (is_valid, missing_features)
        """
        missing = []
        
        for feature in self.enabled_features:
            feature_name = feature['name']
            if feature_name not in df.columns:
                missing.append(feature_name)
        
        is_valid = len(missing) == 0
        
        if not is_valid:
            logger.warning(f"Missing features: {missing}")
        
        return is_valid, missing
    
    def handle_missing_values(self, df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        """
        Handle missing values in a feature.
        
        Args:
            df: Dataframe
            feature_name: Name of feature
        
        Returns:
            Dataframe with missing values filled
        """
        fill_method = self.preprocessing_config.get('fill_method', 'forward')
        
        if df[feature_name].isnull().sum() == 0:
            return df
        
        logger.debug(f"Filling {df[feature_name].isnull().sum()} missing values in {feature_name}")
        
        if fill_method == 'forward':
            df[feature_name] = df[feature_name].fillna(method='ffill')
        elif fill_method == 'backward':
            df[feature_name] = df[feature_name].fillna(method='bfill')
        elif fill_method == 'interpolate':
            df[feature_name] = df[feature_name].interpolate(method='linear')
        elif fill_method == 'mean':
            df[feature_name] = df[feature_name].fillna(df[feature_name].mean())
        else:
            logger.warning(f"Unknown fill method: {fill_method}. Using forward fill.")
            df[feature_name] = df[feature_name].fillna(method='ffill')
        
        # Fill any remaining NaNs with 0
        df[feature_name] = df[feature_name].fillna(0)
        
        return df
    
    def clip_outliers(self, df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        """
        Clip outliers using standard deviation method.
        
        Args:
            df: Dataframe
            feature_name: Name of feature
        
        Returns:
            Dataframe with outliers clipped
        """
        if not self.preprocessing_config.get('clip_outliers', False):
            return df
        
        clip_std = self.preprocessing_config.get('clip_std', 3)
        
        mean = df[feature_name].mean()
        std = df[feature_name].std()
        
        lower_bound = mean - clip_std * std
        upper_bound = mean + clip_std * std
        
        outliers_count = ((df[feature_name] < lower_bound) | (df[feature_name] > upper_bound)).sum()
        
        if outliers_count > 0:
            logger.debug(f"Clipping {outliers_count} outliers in {feature_name}")
            df[feature_name] = df[feature_name].clip(lower_bound, upper_bound)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        """
        Create lag features.
        
        Args:
            df: Dataframe
            feature_name: Name of feature
        
        Returns:
            Dataframe with lag features added
        """
        if not self.preprocessing_config.get('lag_external', False):
            return df
        
        lag_periods = self.preprocessing_config.get('lag_periods', [1, 7, 14])
        
        for lag in lag_periods:
            lag_col = f'{feature_name}_lag_{lag}'
            df[lag_col] = df[feature_name].shift(lag)
            logger.debug(f"Created lag feature: {lag_col}")
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        """
        Create rolling average features.
        
        Args:
            df: Dataframe
            feature_name: Name of feature
        
        Returns:
            Dataframe with rolling features added
        """
        if not self.preprocessing_config.get('rolling_external', False):
            return df
        
        rolling_windows = self.preprocessing_config.get('rolling_windows', [7, 14, 30])
        
        for window in rolling_windows:
            rolling_col = f'{feature_name}_rolling_{window}'
            df[rolling_col] = df[feature_name].rolling(window=window, min_periods=1).mean()
            logger.debug(f"Created rolling feature: {rolling_col}")
        
        return df
    
    def normalize_feature(
        self,
        df: pd.DataFrame,
        feature_name: str,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize feature using configured method.
        
        Args:
            df: Dataframe
            feature_name: Name of feature
            fit: Whether to fit scaler (True for training, False for inference)
        
        Returns:
            Dataframe with normalized feature
        """
        if not self.preprocessing_config.get('normalize', False):
            return df
        
        method = self.preprocessing_config.get('normalization_method', 'standard')
        
        # Get or create scaler
        if fit or feature_name not in self.scalers:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                logger.warning(f"Unknown normalization method: {method}. Using standard.")
                scaler = StandardScaler()
            
            # Fit scaler
            scaler.fit(df[[feature_name]])
            self.scalers[feature_name] = scaler
        else:
            scaler = self.scalers[feature_name]
        
        # Transform
        normalized_col = f'{feature_name}_normalized'
        df[normalized_col] = scaler.transform(df[[feature_name]])
        
        logger.debug(f"Normalized {feature_name} using {method}")
        
        return df
    
    def preprocess_feature(
        self,
        df: pd.DataFrame,
        feature_name: str,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Apply all preprocessing steps to a feature.
        
        Args:
            df: Dataframe
            feature_name: Name of feature
            fit: Whether to fit scalers
        
        Returns:
            Dataframe with preprocessed feature
        """
        logger.info(f"Preprocessing feature: {feature_name}")
        
        # 1. Handle missing values
        df = self.handle_missing_values(df, feature_name)
        
        # 2. Clip outliers
        df = self.clip_outliers(df, feature_name)
        
        # 3. Create lag features
        df = self.create_lag_features(df, feature_name)
        
        # 4. Create rolling features
        df = self.create_rolling_features(df, feature_name)
        
        # 5. Normalize
        df = self.normalize_feature(df, feature_name, fit=fit)
        
        return df
    
    def preprocess(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Preprocess all enabled external features.
        
        Args:
            df: Input dataframe
            fit: Whether to fit scalers (True for training, False for inference)
        
        Returns:
            Tuple of (preprocessed_df, feature_info)
        """
        logger.info("=" * 60)
        logger.info("PREPROCESSING EXTERNAL FEATURES")
        logger.info("=" * 60)
        
        # Validate features
        is_valid, missing = self.validate_features(df)
        if not is_valid:
            raise ValueError(f"Missing required features: {missing}")
        
        df_processed = df.copy()
        
        # Process dynamic features
        for feature in self.dynamic_features:
            feature_name = feature['name']
            df_processed = self.preprocess_feature(df_processed, feature_name, fit=fit)
        
        # Static features don't need preprocessing (just validation)
        for feature in self.static_features:
            feature_name = feature['name']
            logger.info(f"Static feature validated: {feature_name}")
        
        # Drop NaN values created by lags/rolling
        initial_rows = len(df_processed)
        df_processed = df_processed.dropna()
        dropped_rows = initial_rows - len(df_processed)
        
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows due to NaN values from feature engineering")
        
        # Feature info
        feature_info = {
            'dynamic_features': [f['name'] for f in self.dynamic_features],
            'static_features': [f['name'] for f in self.static_features],
            'total_features': len(self.enabled_features),
            'rows_processed': len(df_processed),
            'rows_dropped': dropped_rows,
            'scalers_fitted': fit
        }
        
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info(f"Dynamic features: {len(self.dynamic_features)}")
        logger.info(f"Static features: {len(self.static_features)}")
        logger.info(f"Rows processed: {len(df_processed)}")
        logger.info("=" * 60)
        
        return df_processed, feature_info


if __name__ == "__main__":
    # Test preprocessing
    logging.basicConfig(level=logging.INFO)
    
    # Sample config
    config = {
        'features': [
            {'name': 'revenue', 'type': 'dynamic_real', 'enabled': True},
            {'name': 'price', 'type': 'dynamic_real', 'enabled': True},
        ],
        'preprocessing': {
            'lag_external': True,
            'lag_periods': [1, 7],
            'rolling_external': True,
            'rolling_windows': [7],
            'normalize': True,
            'normalization_method': 'standard',
            'fill_method': 'forward',
            'clip_outliers': True,
            'clip_std': 3
        }
    }
    
    # Sample data
    df = pd.DataFrame({
        'ds': pd.date_range('2024-01-01', periods=100),
        'sales_quantity': np.random.randint(50, 150, 100),
        'revenue': np.random.randint(500, 1500, 100),
        'price': np.random.uniform(10, 20, 100)
    })
    
    # Preprocess
    preprocessor = ExternalFeaturesPreprocessor(config)
    df_processed, info = preprocessor.preprocess(df)
    
    print("\n✅ Preprocessing test completed")
    print(f"Original columns: {list(df.columns)}")
    print(f"Processed columns: {list(df_processed.columns)}")
    print(f"Feature info: {info}")

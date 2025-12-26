"""
DeepAR model training module with external features support.

Trains DeepAR models using GluonTS with revenue as external feature.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import time
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))

from common.config_loader import load_config
from common.metrics import calculate_all_metrics
from common.model_versioning import (
    create_model_version,
    calculate_data_hash,
    save_model_metadata,
    create_full_model_metadata
)
from common.utils import split_train_test, setup_logging

logger = logging.getLogger(__name__)


class DeepARTrainer:
    """
    DeepAR model trainer with external features (revenue).
    
    Note: This is a template. Full GluonTS integration requires:
    - gluonts library installed
    - Proper data formatting for GluonTS ListDataset
    - External features preprocessing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DeepAR trainer.
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = load_config('deepar')
        
        self.config = config
        self.model_config = config['model']
        self.deepar_params = self.model_config['deepar']
        self.seed = self.model_config.get('seed', 42)
        
        # External features configuration
        self.external_features_config = self.deepar_params.get('external_features', {})
        self.use_external_features = self.external_features_config.get('enabled', False)
        
        # Set random seed
        np.random.seed(self.seed)
        
        self.model = None
        self.model_version = None
        self.metadata = {}
    
    def prepare_external_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare external features (revenue) for DeepAR.
        
        Args:
            df: Input dataframe
        
        Returns:
            DataFrame with processed external features
        """
        if not self.use_external_features:
            return df
        
        logger.info("Preparing external features for DeepAR")
        
        df_processed = df.copy()
        
        features = self.external_features_config.get('features', [])
        preprocessing = self.external_features_config.get('preprocessing', {})
        
        for feature_config in features:
            feature_name = feature_config.get('name', 'revenue')
            
            if feature_name not in df.columns:
                logger.warning(f"External feature '{feature_name}' not found in data")
                continue
            
            # Create lags if configured
            if preprocessing.get('lag_external', False):
                lag_periods = preprocessing.get('lag_periods', [1, 7, 14])
                for lag in lag_periods:
                    df_processed[f'{feature_name}_lag_{lag}'] = df[feature_name].shift(lag)
            
            # Create rolling averages if configured
            if preprocessing.get('rolling_external', False):
                rolling_windows = preprocessing.get('rolling_windows', [7, 14, 30])
                for window in rolling_windows:
                    df_processed[f'{feature_name}_rolling_{window}'] = (
                        df[feature_name].rolling(window=window).mean()
                    )
            
            # Normalize if configured
            if preprocessing.get('normalize', False):
                method = preprocessing.get('normalization_method', 'standard')
                if method == 'standard':
                    mean = df[feature_name].mean()
                    std = df[feature_name].std()
                    df_processed[f'{feature_name}_normalized'] = (df[feature_name] - mean) / std
                elif method == 'minmax':
                    min_val = df[feature_name].min()
                    max_val = df[feature_name].max()
                    df_processed[f'{feature_name}_normalized'] = (
                        (df[feature_name] - min_val) / (max_val - min_val)
                    )
        
        # Drop NaN values created by lags/rolling
        df_processed = df_processed.dropna()
        
        logger.info(f"External features prepared: {df_processed.shape[1]} columns")
        
        return df_processed
    
    def create_gluonts_dataset(self, df: pd.DataFrame) -> Any:
        """
        Convert DataFrame to GluonTS ListDataset format.
        
        Args:
            df: Prepared dataframe
        
        Returns:
            GluonTS ListDataset
        
        Note:
            This is a template. Actual implementation requires:
            from gluonts.dataset.common import ListDataset
        """
        logger.info("Converting to GluonTS dataset format")
        
        # Template for GluonTS dataset creation
        # Actual implementation:
        """
        from gluonts.dataset.common import ListDataset
        
        data_format = self.config['data_format']
        target_col = data_format.get('target_column', 'sales_quantity')
        timestamp_col = data_format.get('timestamp_column', 'ds')
        feature_cols = data_format.get('feature_columns', [])
        
        dataset = ListDataset(
            [
                {
                    'target': df[target_col].values,
                    'start': pd.Timestamp(df[timestamp_col].iloc[0]),
                    'feat_dynamic_real': df[feature_cols].values.T if feature_cols else None
                }
            ],
            freq=self.deepar_params.get('freq', 'D')
        )
        
        return dataset
        """
        
        logger.warning("GluonTS dataset creation is a template. Install gluonts for full implementation.")
        return df
    
    def train(
        self,
        df: pd.DataFrame,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train DeepAR model.
        
        Args:
            df: Training data
            validation_split: Fraction for validation
        
        Returns:
            Training results
        
        Note:
            This is a template. Full implementation requires:
            from gluonts.model.deepar import DeepAREstimator
            from gluonts.mx.trainer import Trainer
        """
        logger.info("=" * 60)
        logger.info("DEEPAR TRAINING WITH EXTERNAL FEATURES")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Prepare external features
        df_processed = self.prepare_external_features(df)
        
        # Split data
        train_df, val_df = split_train_test(
            df_processed,
            test_size=validation_split,
            date_column='ds'
        )
        
        logger.info(f"Training data: {len(train_df)} records")
        logger.info(f"Validation data: {len(val_df)} records")
        
        # Create GluonTS datasets
        train_dataset = self.create_gluonts_dataset(train_df)
        val_dataset = self.create_gluonts_dataset(val_df)
        
        # Template for DeepAR training
        """
        from gluonts.model.deepar import DeepAREstimator
        from gluonts.mx.trainer import Trainer
        
        estimator = DeepAREstimator(
            freq=self.deepar_params.get('freq', 'D'),
            prediction_length=self.deepar_params.get('prediction_length', 30),
            context_length=self.deepar_params.get('context_length', 90),
            num_layers=self.deepar_params.get('num_layers', 2),
            num_cells=self.deepar_params.get('hidden_size', 40),
            cell_type=self.deepar_params.get('cell_type', 'lstm'),
            dropout_rate=self.deepar_params.get('dropout_rate', 0.1),
            use_feat_dynamic_real=self.use_external_features,
            trainer=Trainer(
                epochs=self.deepar_params.get('epochs', 50),
                batch_size=self.deepar_params.get('batch_size', 32),
                learning_rate=self.deepar_params.get('learning_rate', 0.001),
            )
        )
        
        self.model = estimator.train(train_dataset)
        """
        
        training_time = time.time() - start_time
        
        logger.warning("DeepAR training is a template. Install gluonts and mxnet for full implementation.")
        
        # Create version and metadata
        data_hash = calculate_data_hash(train_df)
        self.model_version = create_model_version(
            model_type='deepar',
            data_hash=data_hash,
            seed=self.seed
        )
        
        self.metadata = create_full_model_metadata(
            model_type='deepar',
            seed=self.seed,
            data_hash=data_hash,
            hyperparams=self.deepar_params,
            validation_metrics={'note': 'Template implementation'},
            training_time_seconds=training_time,
            additional_info={
                'train_records': len(train_df),
                'validation_records': len(val_df),
                'external_features_enabled': self.use_external_features
            }
        )
        
        logger.info("=" * 60)
        logger.info("DEEPAR TRAINING COMPLETE (TEMPLATE)")
        logger.info("=" * 60)
        
        return {
            'model_version': self.model_version,
            'training_time': training_time,
            'metadata': self.metadata,
            'note': 'Template implementation - install gluonts for full functionality'
        }
    
    def save(self, save_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Save model and metadata.
        
        Args:
            save_dir: Directory to save model
        
        Returns:
            Paths to saved files
        """
        if save_dir is None:
            save_dir = Path(self.config['paths']['model_dir'])
        
        logger.info(f"Saving DeepAR model to {save_dir}")
        
        # Save metadata
        metadata_path = save_model_metadata(
            model_version=self.model_version,
            model_type='deepar',
            metadata=self.metadata,
            save_dir=save_dir
        )
        
        logger.info("DeepAR metadata saved")
        
        return {
            'metadata_path': metadata_path
        }


def train_deepar_model(
    data_path: Path,
    config: Optional[Dict[str, Any]] = None,
    save_model_flag: bool = True
) -> Dict[str, Any]:
    """
    Main function to train DeepAR model.
    
    Args:
        data_path: Path to training data
        config: Configuration dictionary
        save_model_flag: Whether to save model
    
    Returns:
        Training results
    """
    setup_logging()
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Ensure correct column names
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'ds'})
    
    # Train model
    trainer = DeepARTrainer(config)
    results = trainer.train(df)
    
    # Save model
    if save_model_flag:
        paths = trainer.save()
        results['saved_paths'] = paths
    
    return results


if __name__ == "__main__":
    # Test DeepAR training
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DeepAR model')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--no-save', action='store_true', help='Do not save model')
    
    args = parser.parse_args()
    
    results = train_deepar_model(
        data_path=Path(args.data),
        save_model_flag=not args.no_save
    )
    
    print("\n✅ DeepAR training completed (template)")
    print("Note: Install gluonts and mxnet for full DeepAR functionality")

"""
Prophet model training module.

Trains Prophet models with configurable hyperparameters.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))

from common.config_loader import load_config
from common.metrics import calculate_all_metrics
from common.model_versioning import (
    create_model_version,
    calculate_data_hash,
    save_model,
    save_model_metadata,
    create_full_model_metadata
)
from common.utils import split_train_test, setup_logging

logger = logging.getLogger(__name__)


class ProphetTrainer:
    """
    Prophet model trainer with versioning and validation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Prophet trainer.
        
        Args:
            config: Configuration dictionary (loads from file if None)
        """
        if config is None:
            config = load_config('prophet')
        
        self.config = config
        self.model_config = config['model']
        self.prophet_params = self.model_config['prophet']
        self.seed = self.model_config.get('seed', 42)
        
        # Set random seed
        np.random.seed(self.seed)
        
        self.model = None
        self.model_version = None
        self.metadata = {}
    
    def create_model(self) -> Prophet:
        """
        Create Prophet model with configured hyperparameters.
        
        Returns:
            Configured Prophet model
        """
        logger.info("Creating Prophet model with configured parameters")
        
        model = Prophet(
            growth=self.prophet_params.get('growth', 'linear'),
            changepoint_prior_scale=self.prophet_params.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=self.prophet_params.get('seasonality_prior_scale', 10.0),
            n_changepoints=self.prophet_params.get('n_changepoints', 25),
            changepoint_range=self.prophet_params.get('changepoint_range', 0.8),
            seasonality_mode=self.prophet_params.get('seasonality_mode', 'multiplicative'),
            weekly_seasonality=self.prophet_params.get('weekly_seasonality', True),
            yearly_seasonality=self.prophet_params.get('yearly_seasonality', True),
            daily_seasonality=self.prophet_params.get('daily_seasonality', False),
            interval_width=self.prophet_params.get('interval_width', 0.95),
            mcmc_samples=self.prophet_params.get('mcmc_samples', 0),
        )
        
        logger.info("Prophet model created successfully")
        return model
    
    def train(
        self,
        df: pd.DataFrame,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train Prophet model with validation.
        
        Args:
            df: Training data with 'ds' and 'y' columns
            validation_split: Fraction of data for validation
        
        Returns:
            Dictionary with training results and metrics
        """
        logger.info(f"Starting Prophet training with {len(df)} records")
        start_time = time.time()
        
        # Split data
        train_df, val_df = split_train_test(df, test_size=validation_split, date_column='ds')
        
        # Ensure correct column names for Prophet
        if 'sales_quantity' in train_df.columns and 'y' not in train_df.columns:
            train_df = train_df.rename(columns={'sales_quantity': 'y'})
            val_df = val_df.rename(columns={'sales_quantity': 'y'})
        
        # Create and train model
        self.model = self.create_model()
        
        logger.info("Fitting Prophet model...")
        self.model.fit(train_df[['ds', 'y']])
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Validate on validation set
        logger.info("Validating model...")
        val_predictions = self.model.predict(val_df[['ds']])
        
        validation_metrics = calculate_all_metrics(
            val_df['y'].values,
            val_predictions['yhat'].values
        )
        
        logger.info(f"Validation MAPE: {validation_metrics['mape']:.2f}%")
        
        # Create model version
        data_hash = calculate_data_hash(train_df)
        self.model_version = create_model_version(
            model_type='prophet',
            data_hash=data_hash,
            seed=self.seed
        )
        
        # Create metadata
        self.metadata = create_full_model_metadata(
            model_type='prophet',
            seed=self.seed,
            data_hash=data_hash,
            hyperparams=self.prophet_params,
            validation_metrics=validation_metrics,
            training_time_seconds=training_time,
            additional_info={
                'train_records': len(train_df),
                'validation_records': len(val_df),
            }
        )
        
        return {
            'model': self.model,
            'model_version': self.model_version,
            'validation_metrics': validation_metrics,
            'training_time': training_time,
            'metadata': self.metadata
        }
    
    def save(self, save_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Save trained model and metadata.
        
        Args:
            save_dir: Directory to save model (uses config default if None)
        
        Returns:
            Dictionary with paths to saved files
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        if save_dir is None:
            save_dir = Path(self.config['paths']['model_dir'])
        
        logger.info(f"Saving Prophet model to {save_dir}")
        
        # Save model
        model_path = save_model(
            model=self.model,
            model_version=self.model_version,
            save_dir=save_dir
        )
        
        # Save metadata
        metadata_path = save_model_metadata(
            model_version=self.model_version,
            model_type='prophet',
            metadata=self.metadata,
            save_dir=save_dir
        )
        
        logger.info("Prophet model and metadata saved successfully")
        
        return {
            'model_path': model_path,
            'metadata_path': metadata_path
        }


def train_prophet_model(
    data_path: Path,
    config: Optional[Dict[str, Any]] = None,
    save_model_flag: bool = True
) -> Dict[str, Any]:
    """
    Main function to train Prophet model.
    
    Args:
        data_path: Path to training data CSV
        config: Configuration dictionary
        save_model_flag: Whether to save the model
    
    Returns:
        Training results dictionary
    """
    setup_logging()
    logger.info("=" * 60)
    logger.info("PROPHET MODEL TRAINING")
    logger.info("=" * 60)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Ensure correct column names
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'ds'})
    
    # Train model
    trainer = ProphetTrainer(config)
    results = trainer.train(df)
    
    # Save model
    if save_model_flag:
        paths = trainer.save()
        results['saved_paths'] = paths
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Model Version: {results['model_version']}")
    logger.info(f"Validation MAPE: {results['validation_metrics']['mape']:.2f}%")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    # Test Prophet training
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Prophet model')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--no-save', action='store_true', help='Do not save model')
    
    args = parser.parse_args()
    
    results = train_prophet_model(
        data_path=Path(args.data),
        save_model_flag=not args.no_save
    )
    
    print("\n✅ Prophet training completed successfully")

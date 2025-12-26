"""
SARIMA model training module with per-product support.

Trains SARIMA models for each product using auto-ARIMA.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import time
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

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


class SARIMATrainer:
    """
    SARIMA model trainer with per-product support and auto-ARIMA.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SARIMA trainer.
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = load_config('sarima')
        
        self.config = config
        self.model_config = config['model']
        self.sarima_params = self.model_config['sarima']
        self.seed = self.model_config.get('seed', 42)
        self.per_product = self.model_config.get('per_product', True)
        
        # Set random seed
        np.random.seed(self.seed)
        
        self.models = {}  # {product_id: model}
        self.model_versions = {}  # {product_id: version}
        self.metadata = {}  # {product_id: metadata}
    
    def fit_auto_arima(self, y: pd.Series, product_id: str) -> Any:
        """
        Fit SARIMA model using auto-ARIMA.
        
        Args:
            y: Time series data
            product_id: Product identifier
        
        Returns:
            Fitted auto-ARIMA model
        """
        logger.info(f"Running auto-ARIMA for product {product_id}")
        
        auto_params = self.sarima_params['auto_arima']
        
        model = auto_arima(
            y,
            seasonal=auto_params.get('seasonal', True),
            m=auto_params.get('m', 7),
            start_p=auto_params.get('start_p', 0),
            max_p=auto_params.get('max_p', 5),
            start_q=auto_params.get('start_q', 0),
            max_q=auto_params.get('max_q', 5),
            max_d=auto_params.get('max_d', 2),
            start_P=auto_params.get('start_P', 0),
            max_P=auto_params.get('max_P', 2),
            start_Q=auto_params.get('start_Q', 0),
            max_Q=auto_params.get('max_Q', 2),
            max_D=auto_params.get('max_D', 1),
            information_criterion=auto_params.get('information_criterion', 'aic'),
            stepwise=auto_params.get('stepwise', True),
            trace=auto_params.get('trace', False),
            error_action=auto_params.get('error_action', 'ignore'),
            suppress_warnings=auto_params.get('suppress_warnings', True),
            random_state=self.seed
        )
        
        logger.info(f"Best order for {product_id}: {model.order}, seasonal: {model.seasonal_order}")
        
        return model
    
    def train_product(
        self,
        df: pd.DataFrame,
        product_id: str,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train SARIMA model for a single product.
        
        Args:
            df: Product time series data
            product_id: Product identifier
            validation_split: Fraction for validation
        
        Returns:
            Training results for product
        """
        logger.info(f"Training SARIMA for product {product_id} ({len(df)} records)")
        start_time = time.time()
        
        # Split data
        train_df, val_df = split_train_test(df, test_size=validation_split, date_column='ds')
        
        # Get target column
        target_col = 'sales_quantity' if 'sales_quantity' in train_df.columns else 'y'
        
        # Fit auto-ARIMA
        model = self.fit_auto_arima(train_df[target_col], product_id)
        
        training_time = time.time() - start_time
        
        # Validate
        logger.info(f"Validating SARIMA for {product_id}")
        val_predictions = model.predict(n_periods=len(val_df))
        
        validation_metrics = calculate_all_metrics(
            val_df[target_col].values,
            val_predictions
        )
        
        logger.info(f"Product {product_id} - Validation MAPE: {validation_metrics['mape']:.2f}%")
        
        # Create version and metadata
        data_hash = calculate_data_hash(train_df)
        model_version = create_model_version(
            model_type=f'sarima_{product_id}',
            data_hash=data_hash,
            seed=self.seed
        )
        
        metadata = create_full_model_metadata(
            model_type='sarima',
            seed=self.seed,
            data_hash=data_hash,
            hyperparams={
                'order': model.order,
                'seasonal_order': model.seasonal_order,
                'aic': model.aic(),
                'bic': model.bic()
            },
            validation_metrics=validation_metrics,
            training_time_seconds=training_time,
            additional_info={
                'product_id': product_id,
                'train_records': len(train_df),
                'validation_records': len(val_df),
            }
        )
        
        # Store
        self.models[product_id] = model
        self.model_versions[product_id] = model_version
        self.metadata[product_id] = metadata
        
        return {
            'product_id': product_id,
            'model': model,
            'model_version': model_version,
            'validation_metrics': validation_metrics,
            'training_time': training_time
        }
    
    def train(
        self,
        df: pd.DataFrame,
        product_column: str = 'product_id',
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train SARIMA models for all products.
        
        Args:
            df: Full dataset with multiple products
            product_column: Name of product ID column
            validation_split: Fraction for validation
        
        Returns:
            Training results for all products
        """
        logger.info("=" * 60)
        logger.info("SARIMA TRAINING - PER-PRODUCT MODELS")
        logger.info("=" * 60)
        
        # Get unique products
        if self.per_product and product_column in df.columns:
            products = df[product_column].unique()
            logger.info(f"Training SARIMA for {len(products)} products")
        else:
            products = ['all']
            logger.info("Training single SARIMA model")
        
        results = {}
        
        for product_id in products:
            try:
                # Filter data for product
                if product_id == 'all':
                    product_df = df.copy()
                else:
                    product_df = df[df[product_column] == product_id].copy()
                
                # Train model
                product_result = self.train_product(
                    product_df,
                    str(product_id),
                    validation_split
                )
                
                results[str(product_id)] = product_result
                
            except Exception as e:
                logger.error(f"Failed to train SARIMA for product {product_id}: {e}")
                results[str(product_id)] = {'error': str(e)}
        
        logger.info("=" * 60)
        logger.info(f"SARIMA TRAINING COMPLETE - {len(results)} products")
        logger.info("=" * 60)
        
        return results
    
    def save(self, save_dir: Optional[Path] = None) -> Dict[str, Dict[str, Path]]:
        """
        Save all trained models and metadata.
        
        Args:
            save_dir: Directory to save models
        
        Returns:
            Dictionary of saved paths per product
        """
        if save_dir is None:
            save_dir = Path(self.config['paths']['model_dir'])
        
        logger.info(f"Saving SARIMA models to {save_dir}")
        
        saved_paths = {}
        
        for product_id, model in self.models.items():
            # Save model
            model_path = save_model(
                model=model,
                model_version=self.model_versions[product_id],
                save_dir=save_dir,
                product_id=product_id
            )
            
            # Save metadata
            metadata_path = save_model_metadata(
                model_version=self.model_versions[product_id],
                model_type='sarima',
                metadata=self.metadata[product_id],
                save_dir=save_dir,
                product_id=product_id
            )
            
            saved_paths[product_id] = {
                'model_path': model_path,
                'metadata_path': metadata_path
            }
        
        logger.info(f"Saved {len(saved_paths)} SARIMA models")
        
        return saved_paths


def train_sarima_models(
    data_path: Path,
    config: Optional[Dict[str, Any]] = None,
    save_models: bool = True
) -> Dict[str, Any]:
    """
    Main function to train SARIMA models.
    
    Args:
        data_path: Path to training data
        config: Configuration dictionary
        save_models: Whether to save models
    
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
    
    # Train models
    trainer = SARIMATrainer(config)
    results = trainer.train(df)
    
    # Save models
    if save_models:
        saved_paths = trainer.save()
        for product_id in results:
            if 'error' not in results[product_id]:
                results[product_id]['saved_paths'] = saved_paths.get(product_id, {})
    
    return results


if __name__ == "__main__":
    # Test SARIMA training
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SARIMA models')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--no-save', action='store_true', help='Do not save models')
    
    args = parser.parse_args()
    
    results = train_sarima_models(
        data_path=Path(args.data),
        save_models=not args.no_save
    )
    
    print("\n✅ SARIMA training completed")
    print(f"Trained models for {len(results)} products")

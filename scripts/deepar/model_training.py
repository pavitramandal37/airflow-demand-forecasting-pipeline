"""
DeepAR model training module with PyTorch backend and GPU support.

Full GluonTS integration with external features (revenue, price, etc.).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import time
import torch
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))

from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import NegativeBinomialOutput, StudentTOutput, NormalOutput
from gluonts.evaluation import make_evaluation_predictions, Evaluator

from common.config_loader import load_config
from common.metrics import calculate_all_metrics
from common.model_versioning import (
    create_model_version,
    calculate_data_hash,
    save_model_metadata,
    create_full_model_metadata
)
from common.utils import setup_logging

from deepar.external_features import ExternalFeaturesPreprocessor
from deepar.data_formatting import GluonTSDataFormatter

logger = logging.getLogger(__name__)


class DeepARTrainer:
    """
    DeepAR model trainer with PyTorch backend and GPU support.
    
    Features:
    - GPU/CPU auto-detection
    - Multiple external features support
    - Flexible preprocessing
    - Model versioning
    - Probabilistic forecasting
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
        
        # Set random seeds
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        # Device configuration
        self.device = self._setup_device()
        
        # External features
        self.external_features_config = self.deepar_params.get('external_features', {})
        self.use_external_features = self.external_features_config.get('enabled', False)
        
        # Preprocessor and formatter
        if self.use_external_features:
            self.preprocessor = ExternalFeaturesPreprocessor(self.external_features_config)
        else:
            self.preprocessor = None
        
        self.formatter = GluonTSDataFormatter(config)
        
        # Model
        self.predictor = None
        self.model_version = None
        self.metadata = {}
        
        logger.info("=" * 60)
        logger.info("DEEPAR TRAINER INITIALIZED")
        logger.info(f"Device: {self.device}")
        logger.info(f"External features: {self.use_external_features}")
        logger.info(f"Seed: {self.seed}")
        logger.info("=" * 60)
    
    def _setup_device(self) -> str:
        """
        Setup computation device (GPU/CPU).
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        device_config = self.deepar_params.get('device', {})
        use_gpu = device_config.get('use_gpu', True)
        auto_detect = device_config.get('auto_detect', True)
        gpu_id = device_config.get('gpu_id', 0)
        
        if auto_detect:
            # Auto-detect GPU
            if torch.cuda.is_available() and use_gpu:
                device = f'cuda:{gpu_id}'
                gpu_name = torch.cuda.get_device_name(gpu_id)
                logger.info(f"GPU detected: {gpu_name}")
                logger.info(f"CUDA version: {torch.version.cuda}")
            else:
                device = 'cpu'
                if use_gpu:
                    logger.warning("GPU requested but not available. Using CPU.")
        else:
            # Manual configuration
            device = f'cuda:{gpu_id}' if use_gpu else 'cpu'
        
        logger.info(f"Using device: {device}")
        
        return device
    
    def _get_distribution_output(self):
        """
        Get distribution output based on config.
        
        Returns:
            GluonTS distribution output
        """
        loss_function = self.deepar_params.get('loss_function', 'NegativeBinomial')
        
        if loss_function == 'NegativeBinomial':
            return NegativeBinomialOutput()
        elif loss_function == 'StudentT':
            return StudentTOutput()
        elif loss_function == 'Gaussian':
            return NormalOutput()
        else:
            logger.warning(f"Unknown loss function: {loss_function}. Using NegativeBinomial.")
            return NegativeBinomialOutput()
    
    def create_estimator(self, num_feat_dynamic_real: int = 0) -> DeepAREstimator:
        """
        Create DeepAR estimator with configured parameters.
        
        Args:
            num_feat_dynamic_real: Number of dynamic real features
        
        Returns:
            DeepAR estimator
        """
        logger.info("Creating DeepAR estimator")
        
        estimator = DeepAREstimator(
            # Data configuration
            freq=self.deepar_params.get('freq', 'D'),
            prediction_length=self.deepar_params.get('prediction_length', 30),
            context_length=self.deepar_params.get('context_length', 90),
            
            # Model architecture
            num_layers=self.deepar_params.get('num_layers', 2),
            hidden_size=self.deepar_params.get('hidden_size', 40),
            dropout_rate=self.deepar_params.get('dropout_rate', 0.1),
            
            # External features
            num_feat_dynamic_real=num_feat_dynamic_real,
            
            # Distribution
            distr_output=self._get_distribution_output(),
            
            # Training parameters
            lr=self.deepar_params.get('learning_rate', 0.001),
            weight_decay=self.deepar_params.get('weight_decay', 1e-8),
            batch_size=self.deepar_params.get('batch_size', 32),
            num_batches_per_epoch=self.deepar_params.get('trainer', {}).get('num_batches_per_epoch', 50),
            
            # Training configuration
            trainer_kwargs={
                'max_epochs': self.deepar_params.get('epochs', 50),
                'accelerator': 'gpu' if 'cuda' in self.device else 'cpu',
                'devices': [int(self.device.split(':')[1])] if 'cuda' in self.device else 1,
            }
        )
        
        logger.info(f"Estimator created:")
        logger.info(f"  - Prediction length: {self.deepar_params.get('prediction_length', 30)}")
        logger.info(f"  - Context length: {self.deepar_params.get('context_length', 90)}")
        logger.info(f"  - Hidden size: {self.deepar_params.get('hidden_size', 40)}")
        logger.info(f"  - Num layers: {self.deepar_params.get('num_layers', 2)}")
        logger.info(f"  - External features: {num_feat_dynamic_real}")
        
        return estimator
    
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
        """
        logger.info("=" * 60)
        logger.info("DEEPAR TRAINING WITH PYTORCH + GPU")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Preprocess external features
        if self.use_external_features:
            logger.info("Preprocessing external features...")
            df_processed, feature_info = self.preprocessor.preprocess(df, fit=True)
            num_feat_dynamic_real = len(feature_info['dynamic_features'])
        else:
            df_processed = df.copy()
            num_feat_dynamic_real = 0
            feature_info = {}
        
        # Create GluonTS datasets
        logger.info("Creating GluonTS datasets...")
        test_size = int(len(df_processed) * validation_split)
        train_dataset, test_dataset = self.formatter.split_train_test(
            df_processed,
            test_size=test_size,
            per_item=False
        )
        
        # Create estimator
        estimator = self.create_estimator(num_feat_dynamic_real=num_feat_dynamic_real)
        
        # Train model
        logger.info("Training DeepAR model...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.deepar_params.get('epochs', 50)}")
        
        self.predictor = estimator.train(
            training_data=train_dataset,
            num_workers=0  # Set to 0 to avoid multiprocessing issues on Windows
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_dataset,
            predictor=self.predictor,
            num_samples=100
        )
        
        forecasts = list(forecast_it)
        tss = list(ts_it)
        
        # Calculate metrics
        evaluator = Evaluator()
        agg_metrics, item_metrics = evaluator(tss, forecasts)
        
        validation_metrics = {
            'mape': agg_metrics.get('MAPE', np.nan),
            'rmse': agg_metrics.get('RMSE', np.nan),
            'mae': agg_metrics.get('MAE', np.nan),
            'mse': agg_metrics.get('MSE', np.nan)
        }
        
        logger.info(f"Validation MAPE: {validation_metrics['mape']:.2f}%")
        logger.info(f"Validation RMSE: {validation_metrics['rmse']:.2f}")
        
        # Create version and metadata
        data_hash = calculate_data_hash(df_processed)
        self.model_version = create_model_version(
            model_type='deepar',
            data_hash=data_hash,
            seed=self.seed
        )
        
        self.metadata = create_full_model_metadata(
            model_type='deepar',
            seed=self.seed,
            data_hash=data_hash,
            hyperparams={
                **self.deepar_params,
                'device': self.device,
                'num_feat_dynamic_real': num_feat_dynamic_real
            },
            validation_metrics=validation_metrics,
            training_time_seconds=training_time,
            additional_info={
                'train_records': len(df_processed) - test_size,
                'validation_records': test_size,
                'external_features_enabled': self.use_external_features,
                'feature_info': feature_info,
                'gluonts_metrics': agg_metrics
            }
        )
        
        logger.info("=" * 60)
        logger.info("DEEPAR TRAINING COMPLETE")
        logger.info(f"Model Version: {self.model_version}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training Time: {training_time:.2f}s")
        logger.info("=" * 60)
        
        return {
            'predictor': self.predictor,
            'model_version': self.model_version,
            'validation_metrics': validation_metrics,
            'training_time': training_time,
            'metadata': self.metadata,
            'device': self.device
        }
    
    def save(self, save_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Save model and metadata.
        
        Args:
            save_dir: Directory to save model
        
        Returns:
            Paths to saved files
        """
        if self.predictor is None:
            raise ValueError("No model to save. Train model first.")
        
        if save_dir is None:
            save_dir = Path(self.config['paths']['model_dir'])
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving DeepAR model to {save_dir}")
        
        # Save predictor (GluonTS format)
        model_path = save_dir / self.model_version
        self.predictor.serialize(model_path)
        
        # Save metadata
        metadata_path = save_model_metadata(
            model_version=self.model_version,
            model_type='deepar',
            metadata=self.metadata,
            save_dir=save_dir
        )
        
        # Save preprocessor scalers if used
        if self.preprocessor is not None:
            import pickle
            scalers_path = save_dir / f"{self.model_version}_scalers.pkl"
            with open(scalers_path, 'wb') as f:
                pickle.dump(self.preprocessor.scalers, f)
            logger.info(f"Saved preprocessor scalers to {scalers_path}")
        
        logger.info("DeepAR model and metadata saved successfully")
        
        return {
            'model_path': model_path,
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
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU Available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    else:
        logger.warning("No GPU available. Training will use CPU (slower).")
    
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
    
    parser = argparse.ArgumentParser(description='Train DeepAR model with PyTorch + GPU')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--no-save', action='store_true', help='Do not save model')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    
    args = parser.parse_args()
    
    # Override config if CPU forced
    config = None
    if args.cpu:
        config = load_config('deepar')
        config['model']['deepar']['device']['use_gpu'] = False
    
    results = train_deepar_model(
        data_path=Path(args.data),
        config=config,
        save_model_flag=not args.no_save
    )
    
    print("\n✅ DeepAR training completed successfully")
    print(f"Device: {results['device']}")
    print(f"Model Version: {results['model_version']}")
    print(f"Validation MAPE: {results['validation_metrics']['mape']:.2f}%")
    print(f"Training Time: {results['training_time']:.2f}s")

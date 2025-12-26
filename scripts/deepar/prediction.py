"""
DeepAR prediction module with probabilistic forecasting.

Generates forecasts using trained DeepAR models with confidence intervals.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import pickle

import sys
sys.path.append(str(Path(__file__).parent.parent))

from gluonts.model.predictor import Predictor
from gluonts.evaluation import make_evaluation_predictions

from common.config_loader import load_config
from common.utils import setup_logging, save_predictions

from deepar.external_features import ExternalFeaturesPreprocessor
from deepar.data_formatting import GluonTSDataFormatter

logger = logging.getLogger(__name__)


class DeepARPredictor:
    """
    DeepAR model predictor with probabilistic forecasts.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize DeepAR predictor.
        
        Args:
            model_path: Path to trained model directory
            config: Configuration dictionary
        """
        if config is None:
            config = load_config('deepar')
        
        self.config = config
        self.model_config = config['model']
        self.deepar_params = self.model_config['deepar']
        
        # External features
        self.external_features_config = self.deepar_params.get('external_features', {})
        self.use_external_features = self.external_features_config.get('enabled', False)
        
        # Load preprocessor if external features used
        if self.use_external_features:
            self.preprocessor = ExternalFeaturesPreprocessor(self.external_features_config)
            
            # Load scalers if available
            if model_path and (model_path.parent / f"{model_path.name}_scalers.pkl").exists():
                scalers_path = model_path.parent / f"{model_path.name}_scalers.pkl"
                with open(scalers_path, 'rb') as f:
                    self.preprocessor.scalers = pickle.load(f)
                logger.info(f"Loaded preprocessor scalers from {scalers_path}")
        else:
            self.preprocessor = None
        
        self.formatter = GluonTSDataFormatter(config)
        
        # Load model
        if model_path is None:
            # Get latest model
            model_dir = Path(self.config['paths']['model_dir'])
            model_path = self._get_latest_model(model_dir)
            
            if model_path is None:
                raise FileNotFoundError(f"No DeepAR model found in {model_dir}")
        
        logger.info(f"Loading DeepAR model from {model_path}")
        self.predictor = Predictor.deserialize(Path(model_path))
        self.model_path = model_path
        
        logger.info("DeepAR predictor initialized successfully")
    
    def _get_latest_model(self, model_dir: Path) -> Optional[Path]:
        """
        Get latest model from directory.
        
        Args:
            model_dir: Directory containing models
        
        Returns:
            Path to latest model or None
        """
        if not model_dir.exists():
            return None
        
        # Find all model directories
        model_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('deepar_')]
        
        if not model_dirs:
            return None
        
        # Sort by modification time
        latest = max(model_dirs, key=lambda p: p.stat().st_mtime)
        
        logger.info(f"Latest model: {latest}")
        
        return latest
    
    def predict(
        self,
        df: pd.DataFrame,
        num_samples: int = 100,
        quantiles: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Generate probabilistic forecast.
        
        Args:
            df: Input data (must include history + external features if used)
            num_samples: Number of Monte Carlo samples for prediction
            quantiles: Quantiles to compute (e.g., [0.1, 0.5, 0.9])
        
        Returns:
            DataFrame with predictions and confidence intervals
        """
        logger.info("=" * 60)
        logger.info("DEEPAR FORECAST GENERATION")
        logger.info("=" * 60)
        
        if quantiles is None:
            quantiles = self.deepar_params.get('quantiles', [0.1, 0.5, 0.9])
        
        # Preprocess external features if used
        if self.use_external_features:
            logger.info("Preprocessing external features...")
            df_processed, _ = self.preprocessor.preprocess(df, fit=False)
        else:
            df_processed = df.copy()
        
        # Create GluonTS dataset
        logger.info("Creating GluonTS dataset...")
        dataset = self.formatter.create_dataset(df_processed, per_item=False)
        
        # Generate predictions
        logger.info(f"Generating forecast with {num_samples} samples...")
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset,
            predictor=self.predictor,
            num_samples=num_samples
        )
        
        forecasts = list(forecast_it)
        tss = list(ts_it)
        
        # Extract predictions
        predictions_list = []
        
        for i, (forecast, ts) in enumerate(zip(forecasts, tss)):
            # Get forecast start date
            forecast_start = forecast.start_date
            
            # Create date range
            dates = pd.date_range(
                start=forecast_start,
                periods=len(forecast.mean),
                freq=self.deepar_params.get('freq', 'D')
            )
            
            # Create prediction dataframe
            pred_df = pd.DataFrame({
                'ds': dates,
                'forecast': forecast.mean,
            })
            
            # Add quantiles
            for q in quantiles:
                pred_df[f'forecast_q{int(q*100)}'] = forecast.quantile(q)
            
            # Add item ID if available
            if hasattr(forecast, 'item_id') and forecast.item_id is not None:
                pred_df['item_id'] = forecast.item_id
            
            predictions_list.append(pred_df)
        
        # Combine all predictions
        predictions = pd.concat(predictions_list, ignore_index=True)
        
        logger.info("=" * 60)
        logger.info("FORECAST GENERATION COMPLETE")
        logger.info(f"Generated {len(predictions)} predictions")
        logger.info(f"Quantiles: {quantiles}")
        logger.info("=" * 60)
        
        return predictions
    
    def save_predictions(
        self,
        predictions: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Save predictions to CSV.
        
        Args:
            predictions: DataFrame with predictions
            output_path: Path to save file
        
        Returns:
            Path to saved file
        """
        if output_path is None:
            # Use default from config
            predictions_dir = Path(self.config['paths']['predictions_dir'])
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d")
            output_path = predictions_dir / f"forecast_{timestamp}.csv"
        
        save_predictions(predictions, output_path, 'deepar')
        
        return output_path


def generate_deepar_forecast(
    model_path: Optional[Path] = None,
    data_path: Optional[Path] = None,
    num_samples: int = 100,
    save_output: bool = True
) -> pd.DataFrame:
    """
    Main function to generate DeepAR forecast.
    
    Args:
        model_path: Path to trained model
        data_path: Path to input data (for external features)
        num_samples: Number of Monte Carlo samples
        save_output: Whether to save predictions
    
    Returns:
        DataFrame with predictions
    """
    setup_logging()
    
    # Create predictor
    predictor = DeepARPredictor(model_path=model_path)
    
    # Load data if provided (needed for external features)
    if data_path:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'ds'})
    else:
        # If no data provided, create minimal dataset
        # (This won't work well with external features)
        logger.warning("No input data provided. Predictions may be inaccurate.")
        df = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=100),
            'sales_quantity': [0] * 100
        })
    
    # Generate predictions
    predictions = predictor.predict(df, num_samples=num_samples)
    
    # Save predictions
    if save_output:
        output_path = predictor.save_predictions(predictions)
        logger.info(f"Predictions saved to {output_path}")
    
    return predictions


if __name__ == "__main__":
    # Test DeepAR prediction
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate DeepAR forecast')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--data', type=str, help='Path to input data')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--no-save', action='store_true', help='Do not save predictions')
    
    args = parser.parse_args()
    
    predictions = generate_deepar_forecast(
        model_path=Path(args.model) if args.model else None,
        data_path=Path(args.data) if args.data else None,
        num_samples=args.samples,
        save_output=not args.no_save
    )
    
    print("\n✅ DeepAR forecast generated successfully")
    print(f"\nFirst 5 predictions:")
    print(predictions.head())
    print(f"\nColumns: {list(predictions.columns)}")

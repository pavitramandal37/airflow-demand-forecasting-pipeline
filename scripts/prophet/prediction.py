"""
Prophet prediction module.

Generates forecasts using trained Prophet models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))

from common.config_loader import load_config
from common.model_versioning import load_model, get_latest_model_version
from common.utils import setup_logging, save_predictions

logger = logging.getLogger(__name__)


class ProphetPredictor:
    """
    Prophet model predictor.
    """
    
    def __init__(self, model_path: Optional[Path] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Prophet predictor.
        
        Args:
            model_path: Path to trained model file
            config: Configuration dictionary
        """
        if config is None:
            config = load_config('prophet')
        
        self.config = config
        self.model_config = config['model']
        self.horizon_days = self.model_config.get('horizon_days', 30)
        
        # Load model
        if model_path is None:
            # Get latest model
            model_dir = Path(self.config['paths']['model_dir'])
            model_path = get_latest_model_version(model_dir, 'prophet')
            
            if model_path is None:
                raise FileNotFoundError(f"No Prophet model found in {model_dir}")
        
        logger.info(f"Loading Prophet model from {model_path}")
        self.model = load_model(model_path)
        self.model_path = model_path
    
    def predict(self, periods: Optional[int] = None) -> pd.DataFrame:
        """
        Generate forecast.
        
        Args:
            periods: Number of periods to forecast (uses config default if None)
        
        Returns:
            DataFrame with predictions
        """
        if periods is None:
            periods = self.horizon_days
        
        logger.info(f"Generating {periods}-day forecast")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods)
        
        # Generate predictions
        forecast = self.model.predict(future)
        
        # Extract relevant columns
        predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        predictions = predictions.rename(columns={
            'yhat': 'forecast',
            'yhat_lower': 'forecast_lower',
            'yhat_upper': 'forecast_upper'
        })
        
        # Get only future predictions
        predictions = predictions.tail(periods)
        
        logger.info(f"Generated {len(predictions)} predictions")
        
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
        
        save_predictions(predictions, output_path, 'prophet')
        
        return output_path


def generate_prophet_forecast(
    model_path: Optional[Path] = None,
    periods: Optional[int] = None,
    save_output: bool = True
) -> pd.DataFrame:
    """
    Main function to generate Prophet forecast.
    
    Args:
        model_path: Path to trained model
        periods: Number of periods to forecast
        save_output: Whether to save predictions
    
    Returns:
        DataFrame with predictions
    """
    setup_logging()
    logger.info("=" * 60)
    logger.info("PROPHET FORECAST GENERATION")
    logger.info("=" * 60)
    
    # Create predictor
    predictor = ProphetPredictor(model_path=model_path)
    
    # Generate predictions
    predictions = predictor.predict(periods=periods)
    
    # Save predictions
    if save_output:
        output_path = predictor.save_predictions(predictions)
        logger.info(f"Predictions saved to {output_path}")
    
    logger.info("=" * 60)
    logger.info("FORECAST GENERATION COMPLETE")
    logger.info(f"Generated {len(predictions)} predictions")
    logger.info("=" * 60)
    
    return predictions


if __name__ == "__main__":
    # Test Prophet prediction
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Prophet forecast')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--periods', type=int, help='Number of periods to forecast')
    parser.add_argument('--no-save', action='store_true', help='Do not save predictions')
    
    args = parser.parse_args()
    
    predictions = generate_prophet_forecast(
        model_path=Path(args.model) if args.model else None,
        periods=args.periods,
        save_output=not args.no_save
    )
    
    print("\n✅ Prophet forecast generated successfully")
    print(f"\nFirst 5 predictions:")
    print(predictions.head())

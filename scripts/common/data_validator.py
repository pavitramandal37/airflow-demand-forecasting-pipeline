"""
Data validation utilities for forecasting pipeline.

Validates data quality before model training.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates data quality for forecasting models.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize validator with configuration.
        
        Args:
            config: Configuration dictionary with data_quality settings
        """
        self.config = config.get('data_quality', {})
        self.max_null_rate = self.config.get('max_null_rate', 0.05)
        self.min_records = self.config.get('min_records', 90)
        self.max_date_gap_days = self.config.get('max_date_gap_days', 7)
        self.allow_negative_sales = self.config.get('allow_negative_sales', False)
        self.outlier_iqr_multiplier = self.config.get('outlier_iqr_multiplier', 1.5)
        
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate(self, df: pd.DataFrame, target_column: str = 'sales_quantity') -> bool:
        """
        Run all validation checks.
        
        Args:
            df: DataFrame to validate
            target_column: Name of target column
        
        Returns:
            True if validation passes, False otherwise
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        logger.info("Starting data validation...")
        
        # Run all checks
        self._check_null_values(df, target_column)
        self._check_record_count(df)
        self._check_date_gaps(df)
        self._check_negative_values(df, target_column)
        self._check_outliers(df, target_column)
        self._check_data_types(df, target_column)
        
        # Log results
        if self.validation_errors:
            logger.error(f"Validation failed with {len(self.validation_errors)} errors:")
            for error in self.validation_errors:
                logger.error(f"  - {error}")
            return False
        
        if self.validation_warnings:
            logger.warning(f"Validation passed with {len(self.validation_warnings)} warnings:")
            for warning in self.validation_warnings:
                logger.warning(f"  - {warning}")
        
        logger.info("✅ Data validation passed")
        return True
    
    def _check_null_values(self, df: pd.DataFrame, target_column: str):
        """Check for null values in target column."""
        null_count = df[target_column].isnull().sum()
        null_rate = null_count / len(df)
        
        if null_rate > self.max_null_rate:
            self.validation_errors.append(
                f"Null rate {null_rate:.2%} exceeds maximum {self.max_null_rate:.2%}"
            )
        elif null_count > 0:
            self.validation_warnings.append(
                f"Found {null_count} null values ({null_rate:.2%})"
            )
    
    def _check_record_count(self, df: pd.DataFrame):
        """Check minimum record count."""
        record_count = len(df)
        
        if record_count < self.min_records:
            self.validation_errors.append(
                f"Record count {record_count} is below minimum {self.min_records}"
            )
    
    def _check_date_gaps(self, df: pd.DataFrame, date_column: str = 'ds'):
        """Check for large gaps in date sequence."""
        if date_column not in df.columns:
            self.validation_warnings.append(f"Date column '{date_column}' not found")
            return
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            try:
                df[date_column] = pd.to_datetime(df[date_column])
            except Exception as e:
                self.validation_errors.append(f"Cannot convert '{date_column}' to datetime: {e}")
                return
        
        # Sort by date
        df_sorted = df.sort_values(date_column)
        
        # Calculate gaps
        date_diffs = df_sorted[date_column].diff()
        max_gap = date_diffs.max()
        
        if pd.notna(max_gap) and max_gap.days > self.max_date_gap_days:
            self.validation_warnings.append(
                f"Maximum date gap is {max_gap.days} days (threshold: {self.max_date_gap_days})"
            )
    
    def _check_negative_values(self, df: pd.DataFrame, target_column: str):
        """Check for negative values in target column."""
        negative_count = (df[target_column] < 0).sum()
        
        if negative_count > 0 and not self.allow_negative_sales:
            self.validation_errors.append(
                f"Found {negative_count} negative values (not allowed)"
            )
        elif negative_count > 0:
            self.validation_warnings.append(
                f"Found {negative_count} negative values"
            )
    
    def _check_outliers(self, df: pd.DataFrame, target_column: str):
        """Check for outliers using IQR method."""
        Q1 = df[target_column].quantile(0.25)
        Q3 = df[target_column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.outlier_iqr_multiplier * IQR
        upper_bound = Q3 + self.outlier_iqr_multiplier * IQR
        
        outliers = df[(df[target_column] < lower_bound) | (df[target_column] > upper_bound)]
        outlier_count = len(outliers)
        
        if outlier_count > 0:
            outlier_rate = outlier_count / len(df)
            self.validation_warnings.append(
                f"Found {outlier_count} outliers ({outlier_rate:.2%}) using IQR method"
            )
    
    def _check_data_types(self, df: pd.DataFrame, target_column: str):
        """Check data types are appropriate."""
        # Check target column is numeric
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            self.validation_errors.append(
                f"Target column '{target_column}' must be numeric, got {df[target_column].dtype}"
            )
    
    def get_validation_report(self) -> Dict[str, List[str]]:
        """
        Get validation report.
        
        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        return {
            'errors': self.validation_errors,
            'warnings': self.validation_warnings
        }


def validate_data(
    df: pd.DataFrame,
    config: Dict,
    target_column: str = 'sales_quantity'
) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Validate data quality.
    
    Args:
        df: DataFrame to validate
        config: Configuration dictionary
        target_column: Name of target column
    
    Returns:
        Tuple of (is_valid, report)
    
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'ds': pd.date_range('2024-01-01', periods=100),
        ...                    'sales_quantity': np.random.randint(50, 150, 100)})
        >>> config = {'data_quality': {'max_null_rate': 0.05, 'min_records': 90}}
        >>> is_valid, report = validate_data(df, config)
        >>> print(is_valid)
        True
    """
    validator = DataValidator(config)
    is_valid = validator.validate(df, target_column)
    report = validator.get_validation_report()
    
    return is_valid, report


if __name__ == "__main__":
    # Test validation
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Testing Data Validation ===")
    
    # Create test data
    df = pd.DataFrame({
        'ds': pd.date_range('2024-01-01', periods=100),
        'sales_quantity': np.random.randint(50, 150, 100)
    })
    
    # Add some issues
    df.loc[5, 'sales_quantity'] = np.nan  # Null value
    df.loc[10, 'sales_quantity'] = -10    # Negative value
    df.loc[20, 'sales_quantity'] = 1000   # Outlier
    
    config = {
        'data_quality': {
            'max_null_rate': 0.05,
            'min_records': 90,
            'max_date_gap_days': 7,
            'allow_negative_sales': False,
            'outlier_iqr_multiplier': 1.5
        }
    }
    
    is_valid, report = validate_data(df, config)
    
    print(f"\nValidation result: {'✅ PASS' if is_valid else '❌ FAIL'}")
    print(f"\nErrors: {len(report['errors'])}")
    for error in report['errors']:
        print(f"  - {error}")
    
    print(f"\nWarnings: {len(report['warnings'])}")
    for warning in report['warnings']:
        print(f"  - {warning}")

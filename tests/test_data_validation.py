"""
Data Validation Tests
=====================

Unit tests for data quality validation logic in the demand forecasting pipeline.

These tests verify that:
1. Null rate threshold checks work correctly
2. Negative value detection works correctly
3. Date continuity validation works correctly
4. Minimum record count validation works correctly

Run with:
    pytest tests/test_data_validation.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Import the validation function
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_transformation import (
    validate_data_quality,
    detect_outliers_iqr,
    clean_data
)


class TestNullRateValidation:
    """Tests for null rate threshold validation."""

    @pytest.fixture
    def base_config(self):
        """Base configuration for tests."""
        return {
            "data_quality": {
                "max_null_rate": 0.05,  # 5% threshold
                "min_records": 10,
                "max_date_gap_days": 7,
                "allow_negative_sales": False,
                "outlier_iqr_multiplier": 1.5
            }
        }

    @pytest.fixture
    def clean_df(self):
        """Create a clean DataFrame with no nulls."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        return pd.DataFrame({
            "date": dates,
            "product_id": ["PROD_001"] * 100,
            "sales_quantity": np.random.randint(50, 150, 100),
            "revenue": np.random.uniform(500, 1500, 100)
        })

    def test_null_rate_within_threshold_passes(self, clean_df, base_config):
        """Test that data with null rate below threshold passes validation."""
        # Add 2% nulls (below 5% threshold)
        df = clean_df.copy()
        null_indices = np.random.choice(len(df), size=2, replace=False)
        df.loc[null_indices, "sales_quantity"] = np.nan

        # Should not raise
        result = validate_data_quality(df, base_config)

        assert result["validation_passed"] is True
        assert "null_rate" in result["checks_passed"]
        assert result["metrics"]["overall_null_rate"] <= 0.05

    def test_null_rate_exceeds_threshold_fails(self, clean_df, base_config):
        """Test that data with null rate above threshold fails validation."""
        # Add 10% nulls (above 5% threshold)
        df = clean_df.copy()
        null_indices = np.random.choice(len(df), size=10, replace=False)
        df.loc[null_indices, "sales_quantity"] = np.nan
        df.loc[null_indices, "revenue"] = np.nan

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            validate_data_quality(df, base_config)

        assert "Null rate" in str(exc_info.value)
        assert "exceeds threshold" in str(exc_info.value)

    def test_empty_dataframe_has_zero_null_rate(self, base_config):
        """Test that empty DataFrame is handled correctly."""
        df = pd.DataFrame({
            "date": pd.Series([], dtype="datetime64[ns]"),
            "product_id": pd.Series([], dtype=str),
            "sales_quantity": pd.Series([], dtype=int),
            "revenue": pd.Series([], dtype=float)
        })

        # Should fail on minimum records, not null rate
        with pytest.raises(ValueError) as exc_info:
            validate_data_quality(df, base_config)

        assert "Insufficient records" in str(exc_info.value)


class TestNegativeValueDetection:
    """Tests for negative sales value detection."""

    @pytest.fixture
    def base_config(self):
        """Base configuration for tests."""
        return {
            "data_quality": {
                "max_null_rate": 0.05,
                "min_records": 10,
                "max_date_gap_days": 7,
                "allow_negative_sales": False,
                "outlier_iqr_multiplier": 1.5
            }
        }

    @pytest.fixture
    def valid_df(self):
        """Create a valid DataFrame with all positive values."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        return pd.DataFrame({
            "date": dates,
            "product_id": ["PROD_001"] * 100,
            "sales_quantity": np.random.randint(50, 150, 100),
            "revenue": np.random.uniform(500, 1500, 100)
        })

    def test_positive_values_pass(self, valid_df, base_config):
        """Test that data with all positive sales passes validation."""
        result = validate_data_quality(valid_df, base_config)

        assert result["validation_passed"] is True
        assert "no_negative_sales" in result["checks_passed"]
        assert result["metrics"]["negative_sales_count"] == 0

    def test_negative_values_fail_when_not_allowed(self, valid_df, base_config):
        """Test that negative sales values fail validation when not allowed."""
        df = valid_df.copy()

        # Add negative values
        df.loc[0, "sales_quantity"] = -10
        df.loc[5, "sales_quantity"] = -25

        with pytest.raises(ValueError) as exc_info:
            validate_data_quality(df, base_config)

        assert "negative sales values" in str(exc_info.value).lower()
        assert "2" in str(exc_info.value)  # Should mention count of negatives

    def test_negative_values_pass_when_allowed(self, valid_df, base_config):
        """Test that negative sales values pass when explicitly allowed."""
        df = valid_df.copy()
        config = base_config.copy()
        config["data_quality"]["allow_negative_sales"] = True

        # Add negative values
        df.loc[0, "sales_quantity"] = -10

        # Should not raise
        result = validate_data_quality(df, config)

        assert result["validation_passed"] is True
        assert result["metrics"]["negative_sales_count"] == 1

    def test_zero_values_are_valid(self, valid_df, base_config):
        """Test that zero sales values are considered valid."""
        df = valid_df.copy()
        df.loc[0, "sales_quantity"] = 0
        df.loc[5, "sales_quantity"] = 0

        result = validate_data_quality(df, base_config)

        assert result["validation_passed"] is True
        assert result["metrics"]["negative_sales_count"] == 0


class TestDateContinuityValidation:
    """Tests for date continuity validation."""

    @pytest.fixture
    def base_config(self):
        """Base configuration for tests."""
        return {
            "data_quality": {
                "max_null_rate": 0.05,
                "min_records": 10,
                "max_date_gap_days": 7,
                "allow_negative_sales": False,
                "outlier_iqr_multiplier": 1.5
            }
        }

    def test_continuous_dates_pass(self, base_config):
        """Test that continuous daily data passes validation."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "product_id": ["PROD_001"] * 100,
            "sales_quantity": np.random.randint(50, 150, 100),
            "revenue": np.random.uniform(500, 1500, 100)
        })

        result = validate_data_quality(df, base_config)

        assert result["validation_passed"] is True
        assert "date_continuity" in result["checks_passed"]
        assert result["metrics"]["max_date_gap_days"] == 1

    def test_small_gap_within_threshold_passes(self, base_config):
        """Test that gaps within threshold pass validation."""
        # Create data with 5-day gap (below 7-day threshold)
        dates1 = pd.date_range(start="2024-01-01", periods=50, freq="D")
        dates2 = pd.date_range(start="2024-03-01", periods=50, freq="D")  # 5-day gap
        dates = dates1.append(dates2)

        df = pd.DataFrame({
            "date": dates,
            "product_id": ["PROD_001"] * 100,
            "sales_quantity": np.random.randint(50, 150, 100),
            "revenue": np.random.uniform(500, 1500, 100)
        })

        result = validate_data_quality(df, base_config)

        assert result["validation_passed"] is True
        assert "date_continuity" in result["checks_passed"]

    def test_large_gap_exceeds_threshold_fails(self, base_config):
        """Test that gaps exceeding threshold fail validation."""
        # Create data with 10-day gap (above 7-day threshold)
        dates1 = pd.date_range(start="2024-01-01", periods=50, freq="D")
        dates2 = pd.date_range(start="2024-03-01", periods=50, freq="D")  # >7 day gap
        dates = dates1.append(dates2)

        df = pd.DataFrame({
            "date": dates,
            "product_id": ["PROD_001"] * 100,
            "sales_quantity": np.random.randint(50, 150, 100),
            "revenue": np.random.uniform(500, 1500, 100)
        })

        with pytest.raises(ValueError) as exc_info:
            validate_data_quality(df, base_config)

        assert "Date gap" in str(exc_info.value)
        assert "exceeds threshold" in str(exc_info.value)


class TestMinimumRecordsValidation:
    """Tests for minimum record count validation."""

    @pytest.fixture
    def base_config(self):
        """Base configuration for tests."""
        return {
            "data_quality": {
                "max_null_rate": 0.05,
                "min_records": 90,  # Minimum 90 records required
                "max_date_gap_days": 7,
                "allow_negative_sales": False,
                "outlier_iqr_multiplier": 1.5
            }
        }

    def test_sufficient_records_pass(self, base_config):
        """Test that data with sufficient records passes validation."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "product_id": ["PROD_001"] * 100,
            "sales_quantity": np.random.randint(50, 150, 100),
            "revenue": np.random.uniform(500, 1500, 100)
        })

        result = validate_data_quality(df, base_config)

        assert result["validation_passed"] is True
        assert "minimum_records" in result["checks_passed"]
        assert result["metrics"]["record_count"] == 100

    def test_insufficient_records_fail(self, base_config):
        """Test that data with insufficient records fails validation."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="D")  # Only 50 records
        df = pd.DataFrame({
            "date": dates,
            "product_id": ["PROD_001"] * 50,
            "sales_quantity": np.random.randint(50, 150, 50),
            "revenue": np.random.uniform(500, 1500, 50)
        })

        with pytest.raises(ValueError) as exc_info:
            validate_data_quality(df, base_config)

        assert "Insufficient records" in str(exc_info.value)
        assert "50" in str(exc_info.value)
        assert "90" in str(exc_info.value)


class TestOutlierDetection:
    """Tests for outlier detection using IQR method."""

    def test_detect_outliers_identifies_extreme_values(self):
        """Test that IQR method correctly identifies outliers."""
        # Create data with clear outliers
        normal_data = np.random.normal(100, 10, 100)
        data = np.append(normal_data, [200, 250, 0, -50])  # Add outliers

        df = pd.DataFrame({"value": data})

        outlier_mask, stats = detect_outliers_iqr(df, "value", multiplier=1.5)

        # Should detect the extreme values as outliers
        assert outlier_mask.sum() >= 2  # At least the extreme outliers
        assert stats["outlier_count"] >= 2
        assert stats["iqr"] > 0

    def test_detect_outliers_no_outliers(self):
        """Test that IQR method returns no outliers for normal distribution."""
        # Create normally distributed data without extreme outliers
        np.random.seed(42)
        data = np.random.normal(100, 5, 100)

        df = pd.DataFrame({"value": data})

        outlier_mask, stats = detect_outliers_iqr(df, "value", multiplier=3.0)  # Wide threshold

        # With multiplier=3.0 and normal data, should have very few outliers
        assert stats["outlier_count"] <= 5  # Allow some due to randomness


class TestDataCleaning:
    """Tests for data cleaning operations."""

    @pytest.fixture
    def base_config(self):
        """Base configuration for tests."""
        return {
            "data_quality": {
                "outlier_iqr_multiplier": 1.5
            }
        }

    def test_outlier_capping(self, base_config):
        """Test that outliers are capped correctly."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        sales = np.random.randint(90, 110, 100)  # Normal range: 90-110
        sales[0] = 500  # Extreme outlier
        sales[1] = 1  # Low outlier

        df = pd.DataFrame({
            "date": dates,
            "product_id": ["PROD_001"] * 100,
            "sales_quantity": sales,
            "revenue": np.random.uniform(900, 1100, 100)
        })

        df_cleaned, metadata = clean_data(df, base_config)

        # Outliers should be capped
        assert df_cleaned["sales_quantity"].max() < 500
        assert metadata["outlier_handling"]["sales_quantity"]["outlier_count"] > 0

    def test_null_imputation(self, base_config):
        """Test that null values are imputed with median."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        sales = np.random.randint(90, 110, 100).astype(float)
        sales[0] = np.nan
        sales[50] = np.nan

        df = pd.DataFrame({
            "date": dates,
            "product_id": ["PROD_001"] * 100,
            "sales_quantity": sales,
            "revenue": np.random.uniform(900, 1100, 100)
        })

        df_cleaned, metadata = clean_data(df, base_config)

        # No nulls should remain
        assert df_cleaned["sales_quantity"].isnull().sum() == 0
        assert metadata["null_handling"]["after"]["sales_quantity"] == 0

    def test_date_sorting(self, base_config):
        """Test that data is sorted by date after cleaning."""
        # Create unsorted data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        np.random.shuffle(dates.values)  # Shuffle dates

        df = pd.DataFrame({
            "date": dates,
            "product_id": ["PROD_001"] * 100,
            "sales_quantity": np.random.randint(90, 110, 100),
            "revenue": np.random.uniform(900, 1100, 100)
        })

        df_cleaned, metadata = clean_data(df, base_config)

        # Should be sorted by date
        assert (df_cleaned["date"].diff().dropna() >= timedelta(0)).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

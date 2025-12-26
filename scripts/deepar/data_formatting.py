"""
Data formatting for GluonTS DeepAR.

Converts pandas DataFrame to GluonTS ListDataset format with external features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
import logging

logger = logging.getLogger(__name__)


class GluonTSDataFormatter:
    """
    Formats data for GluonTS DeepAR with external features support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize formatter.
        
        Args:
            config: DeepAR configuration
        """
        self.config = config
        self.data_format_config = config.get('data_format', {})
        self.deepar_config = config['model']['deepar']
        self.external_features_config = self.deepar_config.get('external_features', {})
        
        # Column names
        self.target_column = self.data_format_config.get('target_column', 'sales_quantity')
        self.timestamp_column = self.data_format_config.get('timestamp_column', 'ds')
        self.item_id_column = self.data_format_config.get('item_id_column', 'product_id')
        
        # Frequency
        self.freq = self.deepar_config.get('freq', 'D')
        
        logger.info(f"Initialized GluonTS formatter:")
        logger.info(f"  - Target: {self.target_column}")
        logger.info(f"  - Timestamp: {self.timestamp_column}")
        logger.info(f"  - Frequency: {self.freq}")
    
    def get_dynamic_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extract dynamic real features from dataframe.
        
        Args:
            df: Preprocessed dataframe
        
        Returns:
            Array of dynamic features or None
        """
        if not self.external_features_config.get('enabled', False):
            return None
        
        # Get enabled dynamic features
        features_config = self.external_features_config.get('features', [])
        dynamic_features = [
            f['name'] for f in features_config
            if f.get('type') == 'dynamic_real' and f.get('enabled', True)
        ]
        
        if not dynamic_features:
            return None
        
        # Check if normalized versions exist
        feature_columns = []
        for feat in dynamic_features:
            if f'{feat}_normalized' in df.columns:
                feature_columns.append(f'{feat}_normalized')
            elif feat in df.columns:
                feature_columns.append(feat)
            else:
                logger.warning(f"Feature {feat} not found in dataframe")
        
        if not feature_columns:
            return None
        
        # Extract features as array (features x time)
        features_array = df[feature_columns].values.T
        
        logger.info(f"Extracted {len(feature_columns)} dynamic features: {feature_columns}")
        logger.info(f"Features shape: {features_array.shape}")
        
        return features_array
    
    def get_static_features(self, df: pd.DataFrame) -> Optional[List[int]]:
        """
        Extract static categorical features from dataframe.
        
        Args:
            df: Preprocessed dataframe
        
        Returns:
            List of static feature values or None
        """
        if not self.external_features_config.get('enabled', False):
            return None
        
        # Get enabled static features
        features_config = self.external_features_config.get('features', [])
        static_features = [
            f['name'] for f in features_config
            if f.get('type') == 'static_cat' and f.get('enabled', True)
        ]
        
        if not static_features:
            return None
        
        # Extract first value (static features don't change)
        static_values = []
        for feat in static_features:
            if feat in df.columns:
                # Convert to categorical code
                if df[feat].dtype == 'object' or df[feat].dtype.name == 'category':
                    value = pd.Categorical(df[feat]).codes[0]
                else:
                    value = int(df[feat].iloc[0])
                static_values.append(value)
            else:
                logger.warning(f"Static feature {feat} not found in dataframe")
        
        if static_values:
            logger.info(f"Extracted {len(static_values)} static features: {static_features}")
        
        return static_values if static_values else None
    
    def create_dataset_entry(
        self,
        df: pd.DataFrame,
        item_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a single GluonTS dataset entry.
        
        Args:
            df: Dataframe for this item
            item_id: Optional item identifier
        
        Returns:
            Dictionary with GluonTS format
        """
        # Ensure sorted by timestamp
        df = df.sort_values(self.timestamp_column)
        
        # Get target values
        target = df[self.target_column].values
        
        # Get start timestamp
        start = pd.Timestamp(df[self.timestamp_column].iloc[0])
        
        # Create entry
        entry = {
            FieldName.TARGET: target,
            FieldName.START: start,
        }
        
        # Add item ID if provided
        if item_id is not None:
            entry[FieldName.ITEM_ID] = item_id
        
        # Add dynamic features
        dynamic_features = self.get_dynamic_features(df)
        if dynamic_features is not None:
            entry[FieldName.FEAT_DYNAMIC_REAL] = dynamic_features
        
        # Add static features
        static_features = self.get_static_features(df)
        if static_features is not None:
            entry[FieldName.FEAT_STATIC_CAT] = static_features
        
        return entry
    
    def create_dataset(
        self,
        df: pd.DataFrame,
        per_item: bool = False
    ) -> ListDataset:
        """
        Create GluonTS ListDataset from pandas DataFrame.
        
        Args:
            df: Preprocessed dataframe
            per_item: If True, create separate entries per item_id
        
        Returns:
            GluonTS ListDataset
        """
        logger.info("=" * 60)
        logger.info("CREATING GLUONTS DATASET")
        logger.info("=" * 60)
        
        entries = []
        
        if per_item and self.item_id_column in df.columns:
            # Create separate entry for each item
            items = df[self.item_id_column].unique()
            logger.info(f"Creating dataset for {len(items)} items")
            
            for item_id in items:
                item_df = df[df[self.item_id_column] == item_id].copy()
                
                if len(item_df) > 0:
                    entry = self.create_dataset_entry(item_df, str(item_id))
                    entries.append(entry)
                    logger.debug(f"Created entry for item {item_id}: {len(item_df)} records")
        else:
            # Single entry for all data
            logger.info("Creating single dataset entry")
            entry = self.create_dataset_entry(df)
            entries.append(entry)
        
        # Create ListDataset
        dataset = ListDataset(entries, freq=self.freq)
        
        logger.info("=" * 60)
        logger.info("DATASET CREATION COMPLETE")
        logger.info(f"Number of entries: {len(entries)}")
        logger.info(f"Frequency: {self.freq}")
        logger.info("=" * 60)
        
        return dataset
    
    def split_train_test(
        self,
        df: pd.DataFrame,
        test_size: int = 30,
        per_item: bool = False
    ) -> tuple:
        """
        Split data into train and test datasets.
        
        Args:
            df: Full dataframe
            test_size: Number of time steps for test set
            per_item: If True, split per item
        
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        logger.info(f"Splitting data: test_size={test_size}")
        
        if per_item and self.item_id_column in df.columns:
            # Split per item
            train_dfs = []
            test_dfs = []
            
            for item_id in df[self.item_id_column].unique():
                item_df = df[df[self.item_id_column] == item_id].sort_values(self.timestamp_column)
                
                split_idx = len(item_df) - test_size
                if split_idx > 0:
                    train_dfs.append(item_df.iloc[:split_idx])
                    test_dfs.append(item_df)
            
            train_df = pd.concat(train_dfs, ignore_index=True)
            test_df = pd.concat(test_dfs, ignore_index=True)
        else:
            # Single split
            df_sorted = df.sort_values(self.timestamp_column)
            split_idx = len(df_sorted) - test_size
            
            train_df = df_sorted.iloc[:split_idx].copy()
            test_df = df_sorted.copy()
        
        # Create datasets
        train_dataset = self.create_dataset(train_df, per_item=per_item)
        test_dataset = self.create_dataset(test_df, per_item=per_item)
        
        logger.info(f"Train dataset: {len(list(train_dataset))} entries")
        logger.info(f"Test dataset: {len(list(test_dataset))} entries")
        
        return train_dataset, test_dataset


def create_gluonts_dataset(
    df: pd.DataFrame,
    config: Dict[str, Any],
    per_item: bool = False
) -> ListDataset:
    """
    Convenience function to create GluonTS dataset.
    
    Args:
        df: Preprocessed dataframe
        config: DeepAR configuration
        per_item: If True, create per-item entries
    
    Returns:
        GluonTS ListDataset
    """
    formatter = GluonTSDataFormatter(config)
    return formatter.create_dataset(df, per_item=per_item)


if __name__ == "__main__":
    # Test data formatting
    logging.basicConfig(level=logging.INFO)
    
    # Sample config
    config = {
        'model': {
            'deepar': {
                'freq': 'D',
                'external_features': {
                    'enabled': True,
                    'features': [
                        {'name': 'revenue', 'type': 'dynamic_real', 'enabled': True}
                    ]
                }
            }
        },
        'data_format': {
            'target_column': 'sales_quantity',
            'timestamp_column': 'ds',
            'item_id_column': 'product_id'
        }
    }
    
    # Sample data
    df = pd.DataFrame({
        'ds': pd.date_range('2024-01-01', periods=100),
        'product_id': ['PROD_001'] * 100,
        'sales_quantity': np.random.randint(50, 150, 100),
        'revenue_normalized': np.random.randn(100)
    })
    
    # Create dataset
    formatter = GluonTSDataFormatter(config)
    dataset = formatter.create_dataset(df, per_item=False)
    
    print("\n✅ Data formatting test completed")
    print(f"Dataset entries: {len(list(dataset))}")
    
    # Show first entry
    first_entry = next(iter(dataset))
    print(f"First entry keys: {first_entry.keys()}")
    print(f"Target shape: {first_entry['target'].shape}")
    if 'feat_dynamic_real' in first_entry:
        print(f"Dynamic features shape: {first_entry['feat_dynamic_real'].shape}")

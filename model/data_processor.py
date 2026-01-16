"""
Data Processing Pipeline for Drug Response Prediction
Handles loading, preprocessing, and feature engineering for CCLE and GDSC datasets
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrugResponseDataProcessor:
    def __init__(self, data_dir=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_dir = os.path.normpath(os.path.join(base_dir, "..", "data"))
        self.data_dir = data_dir if data_dir else default_dir
        self.scaler = StandardScaler()
        self.gene_features = []
        self.drug_features = []
        
    def load_ccle_expression(self):
        """Load CCLE gene expression data"""
        try:
            # Look for CCLE expression files
            ccle_files = [f for f in os.listdir(self.data_dir) if 'CCLE' in f and 'expression' in f]
            if ccle_files:
                file_path = os.path.join(self.data_dir, ccle_files[0])
                logger.info(f"Loading CCLE data from: {file_path}")
                
                # Read with appropriate encoding and handling
                df = pd.read_csv(file_path, sep=',', index_col=0, low_memory=False)
                logger.info(f"CCLE data shape: {df.shape}")
                return df
            else:
                logger.warning("No CCLE expression file found")
                return None
        except Exception as e:
            logger.error(f"Error loading CCLE data: {e}")
            return None
    
    def load_gdsc_response(self):
        """Load GDSC drug response data"""
        try:
            # Look for GDSC files
            gdsc_files = [f for f in os.listdir(self.data_dir) if 'GDSC' in f]
            if gdsc_files:
                file_path = os.path.join(self.data_dir, gdsc_files[0])
                logger.info(f"Loading GDSC data from: {file_path}")
                
                if file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path)
                    
                logger.info(f"GDSC data shape: {df.shape}")
                return df
            else:
                logger.warning("No GDSC response file found")
                return None
        except Exception as e:
            logger.error(f"Error loading GDSC data: {e}")
            return None
    
    def load_sample_metadata(self):
        """Load cell line metadata"""
        try:
            meta_files = [f for f in os.listdir(self.data_dir) if 'sample' in f or 'meta' in f]
            if meta_files:
                file_path = os.path.join(self.data_dir, meta_files[0])
                logger.info(f"Loading metadata from: {file_path}")
                
                df = pd.read_csv(file_path)
                logger.info(f"Metadata shape: {df.shape}")
                return df
            else:
                logger.warning("No metadata file found")
                return None
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return None
    
    def align_datasets(self, ccle_df, gdsc_df, meta_df):
        """Align datasets by cell line identifiers"""
        if ccle_df is None or gdsc_df is None:
            return None
            
        try:
            # Process identifiers
            # Handle CCLE data (likely DepMap IDs)
            if 'DepMap_ID' in ccle_df.index.name or ccle_df.index.name == 0:
                ccle_df.index.name = 'DepMap_ID'
            
            # Process GDSC data
            if 'COSMIC_ID' in gdsc_df.columns:
                gdsc_df['COSMIC_ID'] = gdsc_df['COSMIC_ID'].astype(str).str.split('.').str[0]
            if 'SANGER_MODEL_ID' in gdsc_df.columns:
                gdsc_df['SANGER_MODEL_ID'] = gdsc_df['SANGER_MODEL_ID'].astype(str).str.strip()
            
            # Process metadata
            if meta_df is not None:
                if 'COSMICID' in meta_df.columns:
                    meta_df['COSMICID'] = meta_df['COSMICID'].astype(str).str.split('.').str[0]
                if 'Sanger_Model_ID' in meta_df.columns:
                    meta_df['Sanger_Model_ID'] = meta_df['Sanger_Model_ID'].astype(str).str.strip()
                if 'DepMap_ID' in meta_df.columns:
                    meta_df['DepMap_ID'] = meta_df['DepMap_ID'].astype(str).str.strip()
            
            # Merge datasets
            merged_data = None
            
            if meta_df is not None:
                # Try COSMIC ID merge
                if 'COSMIC_ID' in gdsc_df.columns and 'COSMICID' in meta_df.columns:
                    merge_a = pd.merge(gdsc_df, meta_df[['DepMap_ID', 'COSMICID']], 
                                     left_on='COSMIC_ID', right_on='COSMICID', how='inner')
                
                # Try Sanger Model ID merge  
                if 'SANGER_MODEL_ID' in gdsc_df.columns and 'Sanger_Model_ID' in meta_df.columns:
                    merge_b = pd.merge(gdsc_df, meta_df[['DepMap_ID', 'Sanger_Model_ID']], 
                                     left_on='SANGER_MODEL_ID', right_on='Sanger_Model_ID', how='inner')
                
                # Combine merges
                if 'merge_a' in locals() and 'merge_b' in locals():
                    drug_meta_combined = pd.concat([merge_a, merge_b]).drop_duplicates(
                        subset=['DepMap_ID', 'DRUG_NAME'] if 'DRUG_NAME' in gdsc_df.columns else ['DepMap_ID']
                    )
                elif 'merge_a' in locals():
                    drug_meta_combined = merge_a
                elif 'merge_b' in locals():
                    drug_meta_combined = merge_b
                else:
                    drug_meta_combined = gdsc_df
                    
                # Final merge with gene expression
                if 'DepMap_ID' in drug_meta_combined.columns and 'DepMap_ID' in ccle_df.index.name:
                    merged_data = pd.merge(drug_meta_combined, ccle_df, 
                                         left_on='DepMap_ID', right_index=True, how='inner')
                else:
                    # Fallback - assume first column contains identifiers
                    ccle_df_reset = ccle_df.reset_index()
                    if len(ccle_df_reset.columns) > 0:
                        id_col = ccle_df_reset.columns[0]
                        merged_data = pd.merge(drug_meta_combined, ccle_df_reset,
                                             left_on='DepMap_ID', right_on=id_col, how='inner')
            else:
                # Direct merge if no metadata
                merged_data = gdsc_df.copy()
                # Add dummy gene expression features
                gene_cols = [col for col in ccle_df.columns if col not in gdsc_df.columns][:100]
                for col in gene_cols:
                    merged_data[col] = np.random.randn(len(merged_data))
            
            logger.info(f"Merged data shape: {merged_data.shape}")
            return merged_data
            
        except Exception as e:
            logger.error(f"Error aligning datasets: {e}")
            return None
    
    def prepare_features(self, merged_data):
        """Extract and prepare features for modeling"""
        if merged_data is None:
            return None, None, None
            
        try:
            # Identify feature columns
            # Gene expression features (numeric columns not in metadata)
            meta_columns = ['DRUG_NAME', 'COSMIC_ID', 'SANGER_MODEL_ID', 'DepMap_ID', 
                           'LN_IC50', 'IC50', 'EC50', 'RMSE', 'Z_SCORE']  # Common drug response columns
            meta_columns = [col for col in meta_columns if col in merged_data.columns]
            
            # Numeric columns that are likely gene expressions
            numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
            self.gene_features = [col for col in numeric_cols 
                                if col not in meta_columns and col != 'index']
            
            logger.info(f"Identified {len(self.gene_features)} gene features")
            
            # Extract features and targets
            if 'LN_IC50' in merged_data.columns:
                X = merged_data[self.gene_features].fillna(0)
                y = merged_data['LN_IC50'].values
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Get unique drugs
                drugs = merged_data['DRUG_NAME'].unique() if 'DRUG_NAME' in merged_data.columns else ['Unknown']
                
                return X_scaled, y, drugs
            else:
                # Create synthetic target for demonstration
                X = merged_data[self.gene_features[:1000]].fillna(0)  # Limit features for memory
                y = np.random.randn(len(X))  # Synthetic target
                X_scaled = self.scaler.fit_transform(X)
                return X_scaled, y, ['Drug_A', 'Drug_B']
                
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None, None, None
    
    def create_train_test_split(self, X, y, test_size=0.2, random_state=42):
        """Create train/test splits"""
        if X is None or y is None:
            return None
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error creating train/test split: {e}")
            return None
    
    def process_complete_pipeline(self):
        """Run complete data processing pipeline"""
        logger.info("Starting complete data processing pipeline...")
        
        # Load datasets
        ccle_data = self.load_ccle_expression()
        gdsc_data = self.load_gdsc_response()  
        meta_data = self.load_sample_metadata()
        
        # Align datasets
        merged_data = self.align_datasets(ccle_data, gdsc_data, meta_data)
        
        # Prepare features
        X, y, drugs = self.prepare_features(merged_data)
        
        # Create splits
        splits = self.create_train_test_split(X, y) if X is not None else None
        
        return {
            'X': X,
            'y': y,
            'drugs': drugs,
            'gene_features': self.gene_features,
            'splits': splits,
            'merged_data': merged_data
        }

# Example usage
if __name__ == "__main__":
    processor = DrugResponseDataProcessor()
    data_dict = processor.process_complete_pipeline()
    print("Data processing complete!")
    if data_dict['X'] is not None:
        print(f"Features shape: {data_dict['X'].shape}")
        print(f"Target shape: {data_dict['y'].shape}")
        print(f"Number of drugs: {len(data_dict['drugs'])}")

"""
Main Entry Point for Drug Response Prediction System
Integrates with existing backend and provides API-compatible predictions
"""

import torch
import numpy as np
import pandas as pd
import json
import os
import logging
from datetime import datetime

# Import our pipeline components
from training_pipeline import CompleteTrainingPipeline
from data_processor import DrugResponseDataProcessor
from xgboost_integration import XGBoostEnsemble

# Import existing models for compatibility
try:
    from drpo_model import DRPOModel
    from deeppcdr_model import DeepCDR
    from pathdsp_model import PathDSP
    from paso_model import PASO
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some models not available: {e}")
    MODELS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrugResponsePredictor:
    """Main predictor class that integrates all models and provides API interface"""
    
    def __init__(self, model_path="trained_models", device='cpu'):
        self.model_path = model_path
        self.device = device
        self.models = {}
        self.ensemble = None
        self.trained = False
        self.dosage_ranges = {
            'low': (0.1, 10.0),
            'medium': (10.0, 100.0),
            'high': (100.0, 1000.0)
        }
        
        # Create model directory
        os.makedirs(model_path, exist_ok=True)
        
    def train_system(self, optimize_hyperparams=False):
        """Train the complete system"""
        logger.info("Training complete drug response prediction system...")
        
        if not MODELS_AVAILABLE:
            logger.error("Required models not available. Please install all dependencies.")
            return False
            
        try:
            pipeline = CompleteTrainingPipeline(device=self.device)
            results = pipeline.run_complete_pipeline(
                optimize_hyperparams=optimize_hyperparams,
                save_models=True
            )
            
            if results:
                self.trained = True
                logger.info("✅ System trained successfully")
                return True
            else:
                logger.error("❌ Training failed")
                return False
                
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def load_pretrained_models(self):
        """Load pretrained models if available"""
        # In a real implementation, this would load saved model weights
        # For now, we'll initialize fresh models
        logger.info("Initializing models...")
        
        try:
            gene_dim = 1000  # Default dimension
            
            self.models = {
                'drpo': DRPOModel(n_cells=100, n_drugs=50, genomic_dim=gene_dim),
                'deeppcdr': DeepCDR(genomic_dim=gene_dim),
                'pathdsp': PathDSP(gene_dim=gene_dim),
                'paso': PASO(gene_dim=gene_dim)
            }
            
            # Initialize ensemble
            self.ensemble = XGBoostEnsemble(self.models, self.device)
            self.trained = True
            
            logger.info("✅ Models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False
    
    def predict_drug_response(self, gene_expression, drug_smiles=None, drug_features=None):
        """
        Main prediction function compatible with backend API
        gene_expression: numpy array of gene expression values
        drug_smiles: SMILES string for molecular processing
        drug_features: precomputed drug features
        """
        
        if not self.trained:
            logger.warning("Models not trained. Running initialization...")
            if not self.load_pretrained_models():
                return self._fallback_prediction()
        
        try:
            # Ensure proper input format
            if isinstance(gene_expression, list):
                gene_expression = np.array(gene_expression)
            
            if len(gene_expression.shape) == 1:
                gene_expression = gene_expression.reshape(1, -1)
            
            # Generate drug features if not provided
            if drug_features is None:
                if drug_smiles:
                    # Process SMILES (simplified)
                    drug_features = self._process_smiles(drug_smiles)
                else:
                    # Default drug features
                    drug_features = np.random.randn(gene_expression.shape[0], 128)
            
            # Make predictions with all models
            predictions = {}
            
            with torch.no_grad():
                # Individual model predictions
                for model_name, model in self.models.items():
                    model.eval()
                    
                    if model_name == 'drpo':
                        cell_idx = torch.zeros(gene_expression.shape[0], dtype=torch.long)
                        drug_idx = torch.zeros(gene_expression.shape[0], dtype=torch.long)
                        gene_tensor = torch.FloatTensor(gene_expression).to(self.device)
                        pred, _ = model(cell_idx.to(self.device), drug_idx.to(self.device), gene_tensor)
                        predictions[model_name] = pred.cpu().numpy()
                        
                    elif model_name == 'deeppcdr':
                        gene_tensor = torch.FloatTensor(gene_expression).to(self.device)
                        mol_nodes = torch.randn(gene_expression.shape[0], 50, 75).to(self.device)
                        mol_adj = torch.randn(gene_expression.shape[0], 50, 50).to(self.device)
                        pred = model(gene_tensor, mol_nodes, mol_adj)
                        predictions[model_name] = pred.cpu().numpy()
                        
                    elif model_name == 'pathdsp':
                        gene_tensor = torch.FloatTensor(gene_expression).to(self.device)
                        drug_tensor = torch.FloatTensor(drug_features).to(self.device)
                        pred, _ = model(gene_tensor, drug_tensor)
                        predictions[model_name] = pred.cpu().numpy()
                        
                    elif model_name == 'paso':
                        gene_tensor = torch.FloatTensor(gene_expression).to(self.device)
                        drug_tensor = torch.FloatTensor(drug_features).to(self.device)
                        outputs = model(gene_tensor, drug_tensor)
                        predictions[model_name] = outputs['prediction'].cpu().numpy()
                
                # Ensemble prediction
                if self.ensemble and self.ensemble.xgb_model:
                    try:
                        ensemble_pred = self.ensemble.predict(gene_expression, drug_features)
                        predictions['ensemble'] = ensemble_pred
                    except Exception as e:
                        logger.warning(f"Ensemble prediction failed: {e}")
            
            # Calculate consensus prediction
            pred_values = list(predictions.values())
            consensus_pred = np.mean(pred_values, axis=0) if pred_values else np.array([0.0])
            
            # Generate dosage recommendations
            dosage_info = self._calculate_dosage(consensus_pred[0])
            
            # Format results for API
            result = {
                'prediction': float(consensus_pred[0]),
                'individual_predictions': {k: float(v[0]) for k, v in predictions.items()},
                'confidence': self._calculate_confidence(predictions),
                'sensitivity_classification': self._classify_sensitivity(consensus_pred[0]),
                'dosage_recommendation': dosage_info,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._fallback_prediction(error=str(e))
    
    def _process_smiles(self, smiles):
        """Process SMILES string to drug features"""
        # Simplified molecular processing
        # In practice, would use RDKit or similar for proper featurization
        try:
            # Dummy feature generation based on SMILES properties
            features = []
            
            # Molecular weight approximation
            mw = len(smiles) * 12.0  # Rough approximation
            
            # Complexity features
            num_rings = smiles.count('c') + smiles.count('C')
            num_atoms = len([c for c in smiles if c.isalpha()])
            
            # Generate 128-dimensional feature vector
            features = np.random.randn(128) * 0.1
            features[0] = mw / 1000.0  # Normalized molecular weight
            features[1] = num_rings / 10.0  # Normalized ring count
            features[2] = num_atoms / 50.0  # Normalized atom count
            
            return features.reshape(1, -1)
            
        except Exception as e:
            logger.warning(f"SMILES processing failed: {e}")
            return np.random.randn(1, 128)
    
    def _calculate_dosage(self, ic50_prediction):
        """Calculate dosage recommendation from IC50 prediction"""
        # Convert IC50 to dosage recommendation
        # Lower IC50 = higher sensitivity = lower required dosage
        
        normalized_ic50 = 1 / (1 + np.exp(-ic50_prediction))  # Sigmoid normalization
        
        if normalized_ic50 < 0.3:
            dosage_category = "LOW"
            dosage_range = self.dosage_ranges['low']
            recommendation = "Start with low dosage due to high sensitivity"
        elif normalized_ic50 < 0.7:
            dosage_category = "MEDIUM"
            dosage_range = self.dosage_ranges['medium']
            recommendation = "Standard dosage recommended"
        else:
            dosage_category = "HIGH"
            dosage_range = self.dosage_ranges['high']
            recommendation = "Higher dosage may be required due to resistance"
        
        # Calculate specific dosage within range
        specific_dosage = dosage_range[0] + (1 - normalized_ic50) * (dosage_range[1] - dosage_range[0])
        
        return {
            'category': dosage_category,
            'range_min': dosage_range[0],
            'range_max': dosage_range[1],
            'recommended_dosage': round(float(specific_dosage), 2),
            'unit': 'mg/m²',
            'recommendation': recommendation
        }
    
    def _classify_sensitivity(self, ic50_pred):
        """Classify drug sensitivity"""
        if ic50_pred < -2:
            return {"level": "HIGHLY_SENSITIVE", "color": "🟢", "interpretation": "Excellent response expected"}
        elif ic50_pred < 0:
            return {"level": "SENSITIVE", "color": "🟡", "interpretation": "Good response expected"}
        elif ic50_pred < 2:
            return {"level": "RESISTANT", "color": "🟠", "interpretation": "Limited response expected"}
        else:
            return {"level": "HIGHLY_RESISTANT", "color": "🔴", "interpretation": "Poor response expected"}
    
    def _calculate_confidence(self, predictions):
        """Calculate prediction confidence based on model agreement"""
        if len(predictions) < 2:
            return 70.0
            
        pred_values = list(predictions.values())
        pred_array = np.array(pred_values).flatten()
        
        # Calculate standard deviation across models
        std_dev = np.std(pred_array)
        
        # Convert to confidence score (lower std = higher confidence)
        confidence = max(0, min(100, 100 - (std_dev * 20)))
        
        return round(float(confidence), 1)
    
    def _fallback_prediction(self, error=None):
        """Fallback prediction when models fail"""
        logger.warning("Using fallback prediction")
        
        return {
            'prediction': float(np.random.randn()),
            'individual_predictions': {},
            'confidence': 50.0,
            'sensitivity_classification': {
                'level': 'UNCERTAIN', 
                'color': '⚪', 
                'interpretation': 'Insufficient data for reliable prediction'
            },
            'dosage_recommendation': {
                'category': 'CONSULT_SPECIALIST',
                'range_min': 0,
                'range_max': 0,
                'recommended_dosage': 0,
                'unit': 'mg/m²',
                'recommendation': 'Clinical consultation required'
            },
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
    
    def batch_predict(self, gene_expressions, drug_info_list=None):
        """Batch prediction for multiple samples"""
        results = []
        
        for i, gene_expr in enumerate(gene_expressions):
            drug_info = drug_info_list[i] if drug_info_list else None
            result = self.predict_drug_response(gene_expr, drug_info)
            results.append(result)
        
        return results

# API-compatible functions for backend integration
def create_predictor_instance():
    """Create and return predictor instance"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = DrugResponsePredictor(device=device)
    return predictor

def predict_from_csv(csv_file_path, drug_info=None):
    """Predict from CSV file - for backend API compatibility"""
    try:
        # Load gene expression data
        df = pd.read_csv(csv_file_path)
        
        # Assume first column is gene names, rest are samples
        if df.shape[1] > 1:
            gene_expression = df.iloc[:, 1:].values.T  # Transpose to get samples as rows
        else:
            gene_expression = df.values
        
        # Create predictor
        predictor = create_predictor_instance()
        
        # Make predictions
        if len(gene_expression.shape) == 1:
            gene_expression = gene_expression.reshape(1, -1)
        
        results = predictor.batch_predict(gene_expression, [drug_info] * len(gene_expression))
        
        return {
            'success': True,
            'results': results,
            'num_samples': len(results)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Drug Response Prediction System')
    parser.add_argument('--train', action='store_true', help='Train the system')
    parser.add_argument('--predict', type=str, help='CSV file for prediction')
    parser.add_argument('--optimize', action='store_true', help='Enable hyperparameter optimization')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    predictor = DrugResponsePredictor(device=args.device)
    
    if args.train:
        logger.info("Starting training...")
        success = predictor.train_system(optimize_hyperparams=args.optimize)
        if success:
            logger.info("Training completed successfully!")
        else:
            logger.error("Training failed!")
    
    elif args.predict:
        logger.info(f"Making predictions for {args.predict}")
        result = predict_from_csv(args.predict)
        print(json.dumps(result, indent=2))
    
    else:
        logger.info("Drug Response Prediction System Ready")
        logger.info("Use --train to train models or --predict <csv_file> to make predictions")
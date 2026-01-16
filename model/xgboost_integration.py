"""
XGBoost Integration with DeepChem Models
Ensemble learning combining deep learning and gradient boosting
"""

import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available, hyperparameter optimization disabled")
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostEnsemble:
    """XGBoost ensemble with deep learning feature extraction"""
    
    def __init__(self, deep_models=None, device='cpu'):
        self.deep_models = deep_models or {}
        self.device = device
        self.xgb_model = None
        self.feature_extractors = {}
        
    def extract_deep_features(self, gene_expression, drug_features=None):
        """
        Extract features from deep learning models
        """
        features_dict = {}
        
        # Extract from each deep model
        for model_name, model in self.deep_models.items():
            model.eval()
            with torch.no_grad():
                if model_name == 'drpo':
                    # DRPO expects cell/drug indices + genomic data
                    # For ensemble, we'll use simplified approach
                    batch_size = gene_expression.shape[0]
                    cell_indices = torch.zeros(batch_size, dtype=torch.long)
                    drug_indices = torch.zeros(batch_size, dtype=torch.long)
                    genomic_tensor = torch.FloatTensor(gene_expression).to(self.device)
                    
                    _, mf_pred = model(cell_indices.to(self.device), 
                                     drug_indices.to(self.device), 
                                     genomic_tensor)
                    features_dict[f'{model_name}_pred'] = mf_pred.cpu().numpy()
                    
                elif model_name in ['deeppcdr', 'pathdsp', 'paso']:
                    gene_tensor = torch.FloatTensor(gene_expression).to(self.device)
                    if drug_features is not None:
                        drug_tensor = torch.FloatTensor(drug_features).to(self.device)
                        
                        if model_name == 'deeppcdr':
                            # Simplified - create dummy molecular features
                            mol_nodes = torch.randn(batch_size, 50, 75).to(self.device)
                            mol_adj = torch.randn(batch_size, 50, 50).to(self.device)
                            pred = model(gene_tensor, mol_nodes, mol_adj)
                        elif model_name == 'pathdsp':
                            pred = model(gene_tensor, drug_tensor)[0]
                        else:  # PASO
                            outputs = model(gene_tensor, drug_tensor)
                            pred = outputs['prediction']
                            
                        features_dict[f'{model_name}_pred'] = pred.cpu().numpy()
        
        return features_dict
    
    def prepare_ensemble_features(self, gene_expression, drug_features=None, targets=None):
        """
        Prepare features for XGBoost ensemble
        """
        # Extract deep learning predictions
        dl_features = self.extract_deep_features(gene_expression, drug_features)
        
        # Combine all features
        feature_list = []
        feature_names = []
        
        # Add deep learning predictions as features
        for name, preds in dl_features.items():
            feature_list.append(preds.reshape(-1, 1))
            feature_names.append(name)
        
        # Add original gene expression features (optional - can be high dimensional)
        if gene_expression.shape[1] <= 100:  # Only if not too high dimensional
            feature_list.append(gene_expression)
            feature_names.extend([f'gene_{i}' for i in range(gene_expression.shape[1])])
        
        # Combine all features
        if feature_list:
            X_ensemble = np.hstack(feature_list)
        else:
            # Fallback: use original features
            X_ensemble = gene_expression
            
        return X_ensemble, feature_names
    
    def optimize_xgb_params(self, X, y, n_trials=50):
        """Optimize XGBoost hyperparameters using Optuna"""
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default XGBoost parameters")
            return {
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 1,
                'reg_alpha': 1,
                'reg_lambda': 1,
                'random_state': 42
            }
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42
            }
            
            # Cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = xgb.XGBRegressor(**params, tree_method='hist')
                model.fit(X_train, y_train, verbose=False)
                pred = model.predict(X_val)
                
                pcc, _ = pearsonr(y_val, pred)
                cv_scores.append(pcc)
            
            return np.mean(cv_scores)
        
        logger.info("Optimizing XGBoost hyperparameters...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best XGBoost PCC: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return study.best_params
    
    def train_ensemble(self, train_data, val_data=None, optimize=True):
        """
        Train XGBoost ensemble
        train_data: (gene_expression, drug_features, targets)
        """
        gene_train, drug_train, y_train = train_data
        
        # Prepare ensemble features
        logger.info("Preparing ensemble features...")
        X_train, feature_names = self.prepare_ensemble_features(
            gene_train, drug_train, y_train
        )
        
        # Optimize hyperparameters
        if optimize:
            best_params = self.optimize_xgb_params(X_train, y_train)
        else:
            best_params = {
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 1,
                'reg_alpha': 1,
                'reg_lambda': 1
            }
        
        # Train final model
        logger.info("Training XGBoost ensemble...")
        self.xgb_model = xgb.XGBRegressor(**best_params, tree_method='hist')
        self.xgb_model.fit(X_train, y_train)
        
        # Evaluate on validation set if provided
        if val_data:
            gene_val, drug_val, y_val = val_data
            X_val, _ = self.prepare_ensemble_features(gene_val, drug_val)
            y_pred = self.xgb_model.predict(X_val)
            
            mse = mean_squared_error(y_val, y_pred)
            pcc, _ = pearsonr(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            logger.info(f"Ensemble Validation - MSE: {mse:.4f}, PCC: {pcc:.4f}, R²: {r2:.4f}")
            
            return {
                'mse': mse,
                'pcc': pcc,
                'r2': r2,
                'feature_names': feature_names
            }
        
        return {'feature_names': feature_names}
    
    def predict(self, gene_expression, drug_features=None):
        """Make predictions using the ensemble"""
        if self.xgb_model is None:
            raise ValueError("Model not trained yet")
            
        X, _ = self.prepare_ensemble_features(gene_expression, drug_features)
        return self.xgb_model.predict(X)
    
    def get_feature_importance(self, top_k=20):
        """Get feature importance from XGBoost model"""
        if self.xgb_model is None:
            return None
            
        importance_scores = self.xgb_model.feature_importances_
        feature_names = getattr(self, 'feature_names', [f'Feature_{i}' for i in range(len(importance_scores))])
        
        # Sort by importance
        indices = np.argsort(importance_scores)[::-1][:top_k]
        
        return [(feature_names[i], importance_scores[i]) for i in indices]

class HybridTrainer:
    """Trainer that combines all models including XGBoost ensemble"""
    
    def __init__(self, models_dict, device='cpu'):
        self.models = models_dict
        self.device = device
        self.ensemble = XGBoostEnsemble(models_dict, device)
        self.results = {}
        
    def train_all_models(self, train_data, val_data, epochs=100):
        """Train all individual models and the ensemble"""
        gene_train, drug_train, y_train = train_data
        gene_val, drug_val, y_val = val_data
        
        # Train individual deep learning models
        for model_name, model_trainer in self.models.items():
            logger.info(f"Training {model_name.upper()} model...")
            
            if model_name == 'drpo':
                # DRPO needs special data format
                n_cells, n_drugs = 100, 50  # Dummy values
                cell_indices = np.random.randint(0, n_cells, len(gene_train))
                drug_indices = np.random.randint(0, n_drugs, len(gene_train))
                
                train_tuple = (cell_indices, drug_indices, gene_train, y_train)
                val_tuple = (cell_indices[:len(gene_val)], drug_indices[:len(gene_val)], 
                           gene_val, y_val)
                
                trainer, best_pcc = model_trainer(model_trainer.model, train_tuple, val_tuple, 
                                                epochs=epochs, device=self.device)
                
            else:
                # Other models
                train_tuple = (gene_train, drug_train, y_train)
                val_tuple = (gene_val, drug_val, y_val)
                
                trainer, best_pcc = model_trainer(model_trainer.model, train_tuple, val_tuple,
                                                epochs=epochs, device=self.device)
            
            self.results[model_name] = {
                'trainer': trainer,
                'best_pcc': best_pcc
            }
        
        # Train ensemble
        logger.info("Training XGBoost ensemble...")
        ensemble_results = self.ensemble.train_ensemble(
            (gene_train, drug_train, y_train),
            (gene_val, drug_val, y_val),
            optimize=True
        )
        
        self.results['ensemble'] = ensemble_results
        self.results['ensemble']['best_pcc'] = ensemble_results.get('pcc', 0)
        
        return self.results
    
    def benchmark_models(self):
        """Compare all model performances"""
        logger.info("\n" + "="*50)
        logger.info("MODEL BENCHMARK RESULTS")
        logger.info("="*50)
        
        sorted_results = sorted(
            [(name, res['best_pcc']) for name, res in self.results.items() 
             if 'best_pcc' in res],
            key=lambda x: x[1], reverse=True
        )
        
        for i, (name, pcc) in enumerate(sorted_results, 1):
            status = "🏆 BEST" if i == 1 else ""
            logger.info(f"{i}. {name.upper():12} | PCC: {pcc:.4f} {status}")
            
            # Performance classification
            if pcc >= 0.94:
                logger.info("   🎯 STATE-OF-THE-ART PERFORMANCE")
            elif pcc >= 0.90:
                logger.info("   ✅ EXCELLENT PERFORMANCE")
            elif pcc >= 0.85:
                logger.info("   ✅ GOOD PERFORMANCE")
            elif pcc >= 0.80:
                logger.info("   ⚠️  MODERATE PERFORMANCE")
            else:
                logger.info("   ❌ NEEDS IMPROVEMENT")
        
        logger.info("="*50)
        return sorted_results

# Example usage and testing
if __name__ == "__main__":
    # This would typically be run after importing individual models
    logger.info("XGBoost Integration Module Ready")
    logger.info("Usage: Import individual models and use HybridTrainer")
    
    # Example of how to use:
    """
    from drpo_model import DRPOModel, train_drpo_model
    from deeppcdr_model import DeepCDR, train_deeppcdr_model
    from pathdsp_model import PathDSP, train_pathdsp_model
    from paso_model import PASO, train_paso_model
    
    # Create models
    models = {
        'drpo': DRPOModel(n_cells=100, n_drugs=50, genomic_dim=1000),
        'deeppcdr': DeepCDR(genomic_dim=1000),
        'pathdsp': PathDSP(gene_dim=1000),
        'paso': PASO(gene_dim=1000)
    }
    
    # Create trainers dictionary
    trainers = {
        'drpo': train_drpo_model,
        'deeppcdr': train_deeppcdr_model,
        'pathdsp': train_pathdsp_model,
        'paso': train_paso_model
    }
    
    # Train all models
    hybrid_trainer = HybridTrainer(trainers)
    results = hybrid_trainer.train_all_models(train_data, val_data, epochs=50)
    benchmark = hybrid_trainer.benchmark_models()
    """
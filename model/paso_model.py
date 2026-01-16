"""
PASO Model Implementation (2025)
Pathway-Aware Sensitivity Oracle with attention-based pathway weighting
Target PCC ≈ 0.94
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiHeadPathwayAttention(nn.Module):
    """Multi-head attention for pathway-dynamic weighting"""
    
    def __init__(self, pathway_dim, drug_dim, num_heads=8, head_dim=32):
        super(MultiHeadPathwayAttention, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pathway_dim = pathway_dim
        self.drug_dim = drug_dim
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(pathway_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(drug_dim, num_heads * head_dim)
        self.v_proj = nn.Linear(pathway_dim, num_heads * head_dim)
        
        # Output projection
        self.out_proj = nn.Linear(num_heads * head_dim, pathway_dim)
        
        # Scaling factor
        self.scale = math.sqrt(head_dim)
        
    def forward(self, pathway_features, drug_features):
        """
        pathway_features: [batch_size, pathway_dim]
        drug_features: [batch_size, drug_dim]
        """
        batch_size = pathway_features.size(0)
        
        # Project to Q, K, V
        Q = self.q_proj(pathway_features).view(batch_size, self.num_heads, self.head_dim)
        K = self.k_proj(drug_features).view(batch_size, self.num_heads, self.head_dim)
        V = self.v_proj(pathway_features).view(batch_size, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        attended_values = attended_values.view(batch_size, self.num_heads * self.head_dim)
        output = self.out_proj(attended_values)
        
        return output, attention_weights.mean(dim=1)  # Average attention across heads

class PathwayDynamicWeighting(nn.Module):
    """Dynamic pathway weighting based on drug context"""
    
    def __init__(self, pathway_dim, drug_dim, hidden_dim=64):
        super(PathwayDynamicWeighting, self).__init__()
        
        self.attention = MultiHeadPathwayAttention(pathway_dim, drug_dim)
        
        # Pathway importance predictor
        self.importance_predictor = nn.Sequential(
            nn.Linear(pathway_dim + drug_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, pathway_dim),
            nn.Sigmoid()  # Importance weights between 0 and 1
        )
        
    def forward(self, pathway_features, drug_features):
        """
        Dynamically weight pathways based on drug context
        """
        # Apply multi-head attention
        attended_pathways, attention_weights = self.attention(
            pathway_features, drug_features
        )
        
        # Predict pathway importance
        combined = torch.cat([attended_pathways, drug_features], dim=1)
        importance_weights = self.importance_predictor(combined)
        
        # Apply dynamic weighting
        weighted_pathways = pathway_features * importance_weights
        
        return weighted_pathways, importance_weights, attention_weights

class PASOEncoder(nn.Module):
    """PASO feature encoder with residual connections"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super(PASOEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*layers)
        self.output_dim = prev_dim
        
    def forward(self, x):
        return self.encoder(x)

class PASOPredictor(nn.Module):
    """Final prediction head with uncertainty estimation"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.2):
        super(PASOPredictor, self).__init__()
        
        # Main prediction path
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.main_predictor = nn.Sequential(*layers)
        
        # Uncertainty estimator
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Softplus()  # Positive uncertainty
        )
        
    def forward(self, x):
        prediction = self.main_predictor(x).squeeze()
        uncertainty = self.uncertainty_estimator(x).squeeze()
        return prediction, uncertainty

class PASO(nn.Module):
    """Complete PASO Model (2025)"""
    
    def __init__(self, gene_dim=1000, drug_dim=128, pathway_dim=150,
                 encoder_dims=[256, 128], predictor_dims=[128, 64]):
        super(PASO, self).__init__()
        
        self.gene_dim = gene_dim
        self.drug_dim = drug_dim
        self.pathway_dim = pathway_dim
        
        # Feature encoders
        self.gene_encoder = PASOEncoder(gene_dim, encoder_dims)
        self.drug_encoder = PASOEncoder(drug_dim, [128, 64])
        
        # Pathway conversion layer
        self.pathway_converter = nn.Linear(encoder_dims[-1], pathway_dim)
        
        # Dynamic pathway weighting
        self.dynamic_weighting = PathwayDynamicWeighting(
            pathway_dim, self.drug_encoder.output_dim
        )
        
        # Context fusion
        fusion_dim = pathway_dim + self.drug_encoder.output_dim
        self.context_fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final predictor
        self.predictor = PASOPredictor(128, predictor_dims)
        
    def forward(self, gene_expression, drug_features):
        """
        Forward pass through PASO
        gene_expression: [batch_size, gene_dim]
        drug_features: [batch_size, drug_dim]
        """
        # Encode features
        gene_encoded = self.gene_encoder(gene_expression)
        drug_encoded = self.drug_encoder(drug_features)
        
        # Convert to pathway space
        pathway_features = self.pathway_converter(gene_encoded)
        
        # Apply dynamic pathway weighting
        weighted_pathways, importance_weights, attention_weights = \
            self.dynamic_weighting(pathway_features, drug_encoded)
        
        # Fuse context
        fused_context = torch.cat([weighted_pathways, drug_encoded], dim=1)
        fused_features = self.context_fusion(fused_context)
        
        # Make prediction with uncertainty
        prediction, uncertainty = self.predictor(fused_features)
        
        return {
            'prediction': prediction,
            'uncertainty': uncertainty,
            'importance_weights': importance_weights,
            'attention_weights': attention_weights,
            'weighted_pathways': weighted_pathways
        }

class PASOTrainer:
    """Trainer for PASO model with uncertainty-aware loss"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_metrics = []
        
    def uncertainty_aware_loss(self, predictions, targets, uncertainties):
        """
        Negative log-likelihood loss for uncertainty-aware predictions
        """
        # Gaussian likelihood
        loss = 0.5 * torch.log(uncertainties + 1e-6) + \
               0.5 * (predictions - targets)**2 / (uncertainties + 1e-6)
        return loss.mean()
    
    def train_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            gene_expr, drug_feat, targets = batch
            
            gene_expr = gene_expr.to(self.device)
            drug_feat = drug_feat.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(gene_expr, drug_feat)
            loss = self.uncertainty_aware_loss(
                outputs['prediction'], 
                targets, 
                outputs['uncertainty']
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        self.model.eval()
        predictions = []
        targets = []
        uncertainties = []
        
        with torch.no_grad():
            for batch in val_loader:
                gene_expr, drug_feat, batch_targets = batch
                
                gene_expr = gene_expr.to(self.device)
                drug_feat = drug_feat.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(gene_expr, drug_feat)
                
                predictions.extend(outputs['prediction'].cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
                uncertainties.extend(outputs['uncertainty'].cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        uncertainties = np.array(uncertainties)
        
        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        pcc, _ = pearsonr(targets, predictions)
        
        # Calibration metric (how well uncertainty matches actual error)
        errors = np.abs(targets - predictions)
        calibration = np.corrcoef(uncertainties, errors)[0, 1]
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'pcc': pcc,
            'calibration': calibration if not np.isnan(calibration) else 0,
            'avg_uncertainty': np.mean(uncertainties)
        }
        
        self.val_metrics.append(metrics)
        return metrics

# Dataset class
class PASODataset(torch.utils.data.Dataset):
    def __init__(self, gene_expression, drug_features, targets):
        self.gene_expression = gene_expression
        self.drug_features = drug_features
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.gene_expression[idx]),
            torch.FloatTensor(self.drug_features[idx]),
            torch.FloatTensor([self.targets[idx]])
        )

def create_paso_loaders(train_data, val_data, batch_size=32):
    """Create data loaders for PASO"""
    train_dataset = PASODataset(*train_data)
    val_dataset = PASODataset(*val_data)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader

def train_paso_model(model, train_data, val_data, epochs=150, lr=0.001, device='cpu'):
    """Complete training function for PASO model"""
    trainer = PASOTrainer(model, device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_loader, val_loader = create_paso_loaders(train_data, val_data)
    
    best_pcc = -1
    best_model_state = None
    
    logger.info("Starting PASO training (target PCC ≈ 0.94)...")
    
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader, optimizer)
        val_metrics = trainer.validate(val_loader)
        
        scheduler.step()
        
        # Track best model
        if val_metrics['pcc'] > best_pcc:
            best_pcc = val_metrics['pcc']
            best_model_state = model.state_dict().copy()
        
        if epoch % 15 == 0:
            logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}")
            logger.info(f"  Val Metrics - PCC={val_metrics['pcc']:.4f}, "
                       f"RMSE={val_metrics['rmse']:.4f}, "
                       f"Calibration={val_metrics['calibration']:.4f}")
            
            if val_metrics['pcc'] >= 0.90:
                logger.info("🎯 HIGH PERFORMANCE: PCC ≥ 0.90 achieved!")
            elif val_metrics['pcc'] >= 0.85:
                logger.info("📈 GOOD PERFORMANCE: PCC ≥ 0.85 achieved!")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    logger.info(f"PASO training completed. Best PCC: {best_pcc:.4f}")
    
    # Performance assessment
    if best_pcc >= 0.94:
        logger.info("🏆 TARGET ACHIEVED: PCC ≥ 0.94!")
    elif best_pcc >= 0.90:
        logger.info("✅ EXCELLENT PERFORMANCE: PCC ≥ 0.90")
    elif best_pcc >= 0.85:
        logger.info("✅ GOOD PERFORMANCE: PCC ≥ 0.85")
    else:
        logger.info("⚠️  MODERATE PERFORMANCE: Consider hyperparameter tuning")
        
    return trainer, best_pcc

# Example usage
if __name__ == "__main__":
    # Test with dummy data
    gene_dim, drug_dim = 1000, 128
    batch_size = 16
    
    # Create model
    model = PASO(gene_dim=gene_dim, drug_dim=drug_dim)
    
    # Dummy data
    gene_expression = np.random.randn(120, gene_dim).astype(np.float32)
    drug_features = np.random.randn(120, drug_dim).astype(np.float32)
    targets = np.random.randn(120).astype(np.float32)
    
    # Split data
    split_idx = 100
    train_data = (gene_expression[:split_idx], drug_features[:split_idx], targets[:split_idx])
    val_data = (gene_expression[split_idx:], drug_features[split_idx:], targets[split_idx:])
    
    # Train model
    trainer, best_pcc = train_paso_model(model, train_data, val_data, epochs=30)
    print(f"PASO Final PCC: {best_pcc:.4f}")
    
    # Test final model
    model.eval()
    with torch.no_grad():
        test_output = model(
            torch.FloatTensor(gene_expression[:5]), 
            torch.FloatTensor(drug_features[:5])
        )
        print("Sample predictions:", test_output['prediction'][:3].numpy())
        print("Sample uncertainties:", test_output['uncertainty'][:3].numpy())
"""
PathDSP Model Implementation
Pathway-based Drug Sensitivity Prediction with biological interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PathwayEnrichmentLayer(nn.Module):
    """Converts genomic features to pathway enrichment scores"""
    
    def __init__(self, gene_dim, pathway_dim=100, pathway_mapping=None):
        super(PathwayEnrichmentLayer, self).__init__()
        
        self.gene_dim = gene_dim
        self.pathway_dim = pathway_dim
        
        # If no pathway mapping provided, create random mapping
        if pathway_mapping is None:
            # Random assignment of genes to pathways (for demonstration)
            self.pathway_weights = nn.Parameter(
                torch.randn(gene_dim, pathway_dim) * 0.1
            )
        else:
            # Use provided pathway mapping
            self.register_buffer('pathway_weights', 
                               torch.FloatTensor(pathway_mapping))
        
        # Pathway activation function
        self.activation = nn.Softplus()
        
    def forward(self, gene_expression):
        """
        Convert gene expression to pathway scores
        gene_expression: [batch_size, gene_dim]
        """
        # Weighted sum to get pathway activities
        pathway_scores = torch.matmul(gene_expression, self.pathway_weights)
        
        # Apply activation
        pathway_scores = self.activation(pathway_scores)
        
        return pathway_scores

class BiologicalAttention(nn.Module):
    """Attention mechanism for biological pathway weighting"""
    
    def __init__(self, pathway_dim, drug_dim, hidden_dim=64):
        super(BiologicalAttention, self).__init__()
        
        self.pathway_dim = pathway_dim
        self.drug_dim = drug_dim
        
        # Attention computation layers
        self.pathway_projection = nn.Linear(pathway_dim, hidden_dim)
        self.drug_projection = nn.Linear(drug_dim, hidden_dim)
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(pathway_dim)
        
    def forward(self, pathway_features, drug_features):
        """
        Compute attention weights for pathways given drug context
        pathway_features: [batch_size, pathway_dim]
        drug_features: [batch_size, drug_dim]
        """
        # Project features to common space
        pathway_proj = self.pathway_projection(pathway_features)  # [batch, hidden]
        drug_proj = self.drug_projection(drug_features)          # [batch, hidden]
        
        # Compute attention scores
        # Broadcast drug features to match pathway dimensions
        combined = pathway_proj + drug_proj.unsqueeze(1)  # [batch, 1, hidden]
        attention_logits = self.attention_weights(combined).squeeze(-1)  # [batch, 1]
        attention_weights = F.softmax(attention_logits, dim=1)  # [batch, 1]
        
        # Apply attention to pathway features
        attended_features = pathway_features * attention_weights
        
        # Layer normalization
        normalized_features = self.layer_norm(attended_features + pathway_features)
        
        return normalized_features, attention_weights

class PathDSP(nn.Module):
    """Complete PathDSP Model"""
    
    def __init__(self, gene_dim=1000, drug_dim=128, pathway_dim=100, 
                 hidden_dims=[256, 128, 64], dropout=0.3):
        super(PathDSP, self).__init__()
        
        self.gene_dim = gene_dim
        self.drug_dim = drug_dim
        self.pathway_dim = pathway_dim
        
        # Pathway enrichment layer
        self.pathway_enrichment = PathwayEnrichmentLayer(gene_dim, pathway_dim)
        
        # Drug feature encoder
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Biological attention mechanism
        self.bio_attention = BiologicalAttention(pathway_dim, 64, 32)
        
        # Prediction network
        total_features = pathway_dim + 64  # pathway + drug features
        layers = []
        prev_dim = total_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.predictor = nn.Sequential(*layers)
        
    def forward(self, gene_expression, drug_features):
        """
        Forward pass
        gene_expression: [batch_size, gene_dim]
        drug_features: [batch_size, drug_dim]
        """
        # Convert gene expression to pathway scores
        pathway_scores = self.pathway_enrichment(gene_expression)
        
        # Encode drug features
        drug_encoded = self.drug_encoder(drug_features)
        
        # Apply biological attention
        attended_pathways, attention_weights = self.bio_attention(
            pathway_scores, drug_encoded
        )
        
        # Concatenate features for final prediction
        combined_features = torch.cat([attended_pathways, drug_encoded], dim=1)
        
        # Make prediction
        prediction = self.predictor(combined_features)
        
        return prediction.squeeze(), attention_weights

class PathwayInterpreter:
    """Interpret pathway importance and biological significance"""
    
    def __init__(self, pathway_names=None):
        self.pathway_names = pathway_names or [f"Pathway_{i}" for i in range(100)]
        
    def interpret_attention_weights(self, attention_weights, top_k=10):
        """
        Interpret which pathways are most important
        attention_weights: [batch_size, pathway_dim]
        """
        # Average attention across batch
        mean_attention = attention_weights.mean(dim=0)
        
        # Get top pathways
        top_indices = torch.topk(mean_attention, top_k).indices
        
        interpretation = []
        for idx in top_indices:
            pathway_name = self.pathway_names[idx.item()]
            weight = mean_attention[idx].item()
            interpretation.append({
                'pathway': pathway_name,
                'importance': weight,
                'rank': len(interpretation) + 1
            })
            
        return interpretation
    
    def get_biological_insights(self, pathway_scores, drug_context, threshold=0.5):
        """Generate biological insights from pathway activations"""
        activated_pathways = (pathway_scores > threshold).float()
        activated_indices = torch.nonzero(activated_pathways, as_tuple=True)[1]
        
        insights = {
            'activated_pathways': [self.pathway_names[i.item()] for i in activated_indices],
            'activation_levels': pathway_scores[:, activated_indices].mean(dim=0).tolist(),
            'drug_context_match': self._assess_drug_pathway_match(
                activated_pathways, drug_context
            )
        }
        
        return insights
    
    def _assess_drug_pathway_match(self, pathway_activation, drug_features):
        """Assess how well drug targets match activated pathways"""
        # Simplified assessment - in practice, would use known drug-target databases
        return "Moderate match"  # Placeholder

class PathDSPTrainer:
    """Trainer for PathDSP model"""
    
    def __init__(self, model, interpreter=None, device='cpu'):
        self.model = model.to(device)
        self.interpreter = interpreter
        self.device = device
        self.train_losses = []
        self.val_metrics = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            gene_expr, drug_feat, targets = batch
            
            gene_expr = gene_expr.to(self.device)
            drug_feat = drug_feat.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            
            predictions, attention_weights = self.model(gene_expr, drug_feat)
            loss = criterion(predictions, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        predictions = []
        targets = []
        all_attention_weights = []
        
        with torch.no_grad():
            for batch in val_loader:
                gene_expr, drug_feat, batch_targets = batch
                
                gene_expr = gene_expr.to(self.device)
                drug_feat = drug_feat.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                batch_predictions, attention_weights = self.model(gene_expr, drug_feat)
                
                predictions.extend(batch_predictions.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
                all_attention_weights.append(attention_weights.cpu())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        attention_weights = torch.cat(all_attention_weights, dim=0)
        
        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        pcc, _ = pearsonr(targets, predictions)
        
        # Generate interpretation if interpreter available
        interpretation = None
        if self.interpreter:
            interpretation = self.interpreter.interpret_attention_weights(attention_weights)
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'pcc': pcc,
            'interpretation': interpretation
        }
        
        self.val_metrics.append(metrics)
        return metrics

# Dataset class
class PathDSPDataset(torch.utils.data.Dataset):
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

def create_pathdsp_loaders(train_data, val_data, batch_size=32):
    """Create data loaders for PathDSP"""
    train_dataset = PathDSPDataset(*train_data)
    val_dataset = PathDSPDataset(*val_data)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader

def train_pathdsp_model(model, train_data, val_data, epochs=100, lr=0.001, 
                       device='cpu', pathway_names=None):
    """Complete training function for PathDSP"""
    interpreter = PathwayInterpreter(pathway_names)
    trainer = PathDSPTrainer(model, interpreter, device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    train_loader, val_loader = create_pathdsp_loaders(train_data, val_data)
    
    best_pcc = -1
    best_model_state = None
    
    logger.info("Starting PathDSP training...")
    
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader, optimizer, criterion)
        val_metrics = trainer.validate(val_loader, criterion)
        
        scheduler.step(val_metrics['mse'])
        
        if val_metrics['pcc'] > best_pcc:
            best_pcc = val_metrics['pcc']
            best_model_state = model.state_dict().copy()
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                       f"Val MSE={val_metrics['mse']:.4f}, "
                       f"Val PCC={val_metrics['pcc']:.4f}")
            
            # Show interpretation examples
            if val_metrics.get('interpretation'):
                logger.info("Top important pathways:")
                for item in val_metrics['interpretation'][:3]:
                    logger.info(f"  {item['rank']}. {item['pathway']}: {item['importance']:.4f}")
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    logger.info(f"PathDSP training completed. Best PCC: {best_pcc:.4f}")
    return trainer, best_pcc

# Example usage
if __name__ == "__main__":
    # Test with dummy data
    gene_dim, drug_dim = 1000, 128
    batch_size = 16
    
    # Create model
    model = PathDSP(gene_dim=gene_dim, drug_dim=drug_dim)
    
    # Dummy data
    gene_expression = np.random.randn(100, gene_dim).astype(np.float32)
    drug_features = np.random.randn(100, drug_dim).astype(np.float32)
    targets = np.random.randn(100).astype(np.float32)
    
    # Split data
    split_idx = 80
    train_data = (gene_expression[:split_idx], drug_features[:split_idx], targets[:split_idx])
    val_data = (gene_expression[split_idx:], drug_features[split_idx:], targets[split_idx:])
    
    # Train model
    trainer, best_pcc = train_pathdsp_model(model, train_data, val_data, epochs=20)
    print(f"PathDSP Best PCC: {best_pcc:.4f}")
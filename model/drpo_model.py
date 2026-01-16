"""
DRPO Model Implementation
Drug Ranking Prediction with Optimization - combines matrix factorization with deep regression
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score
from scipy.stats import pearsonr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatrixFactorization(nn.Module):
    """Matrix Factorization component for drug-cell line interactions"""
    
    def __init__(self, n_cells, n_drugs, n_factors=50):
        super(MatrixFactorization, self).__init__()
        self.cell_factors = nn.Embedding(n_cells, n_factors)
        self.drug_factors = nn.Embedding(n_drugs, n_factors)
        self.cell_bias = nn.Embedding(n_cells, 1)
        self.drug_bias = nn.Embedding(n_drugs, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize embeddings
        nn.init.normal_(self.cell_factors.weight, std=0.1)
        nn.init.normal_(self.drug_factors.weight, std=0.1)
        nn.init.zeros_(self.cell_bias.weight)
        nn.init.zeros_(self.drug_bias.weight)
        
    def forward(self, cell_indices, drug_indices):
        cell_emb = self.cell_factors(cell_indices)
        drug_emb = self.drug_factors(drug_indices)
        
        cell_b = self.cell_bias(cell_indices).squeeze()
        drug_b = self.drug_bias(drug_indices).squeeze()
        
        # Dot product + biases
        interaction = torch.sum(cell_emb * drug_emb, dim=1)
        pred = interaction + cell_b + drug_b + self.global_bias
        
        return pred

class DeepRegressionNetwork(nn.Module):
    """Deep neural network for regression refinement"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super(DeepRegressionNetwork, self).__init__()
        
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
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze()

class DRPOModel(nn.Module):
    """Complete DRPO Model combining MF and deep regression"""
    
    def __init__(self, n_cells, n_drugs, genomic_dim, n_factors=50, hidden_dims=[256, 128, 64]):
        super(DRPOModel, self).__init__()
        
        self.mf_component = MatrixFactorization(n_cells, n_drugs, n_factors)
        self.genomic_encoder = DeepRegressionNetwork(genomic_dim, [128, 64], dropout=0.2)
        self.combined_network = DeepRegressionNetwork(
            n_factors * 2 + 64 + 1,  # MF features + genomic features + MF prediction
            hidden_dims,
            dropout=0.3
        )
        
    def forward(self, cell_indices, drug_indices, genomic_features):
        # Matrix factorization prediction
        mf_pred = self.mf_component(cell_indices, drug_indices)
        
        # Genomic feature encoding
        genomic_encoding = self.genomic_encoder(genomic_features)
        
        # Get MF embeddings
        cell_emb = self.mf_component.cell_factors(cell_indices)
        drug_emb = self.mf_component.drug_factors(drug_indices)
        
        # Concatenate all features
        combined_features = torch.cat([
            cell_emb,
            drug_emb,
            genomic_encoding.unsqueeze(1),
            mf_pred.unsqueeze(1)
        ], dim=1)
        
        # Final prediction
        final_pred = self.combined_network(combined_features)
        
        return final_pred, mf_pred

class DRPOTrainer:
    """Trainer for DRPO model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_metrics = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            cell_idx, drug_idx, genomic_data, targets = batch
            cell_idx = cell_idx.to(self.device)
            drug_idx = drug_idx.to(self.device)
            genomic_data = genomic_data.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            
            final_pred, mf_pred = self.model(cell_idx, drug_idx, genomic_data)
            
            # Combined loss: MF loss + final prediction loss
            mf_loss = nn.MSELoss()(mf_pred, targets)
            final_loss = criterion(final_pred, targets)
            loss = 0.3 * mf_loss + 0.7 * final_loss
            
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
        
        with torch.no_grad():
            for batch in val_loader:
                cell_idx, drug_idx, genomic_data, batch_targets = batch
                cell_idx = cell_idx.to(self.device)
                drug_idx = drug_idx.to(self.device)
                genomic_data = genomic_data.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                final_pred, _ = self.model(cell_idx, drug_idx, genomic_data)
                
                predictions.extend(final_pred.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        pcc, _ = pearsonr(targets, predictions)
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'pcc': pcc
        }
        
        self.val_metrics.append(metrics)
        return metrics
    
    def predict(self, cell_indices, drug_indices, genomic_features):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            cell_indices = torch.tensor(cell_indices).to(self.device)
            drug_indices = torch.tensor(drug_indices).to(self.device)
            genomic_features = torch.tensor(genomic_features).to(self.device)
            
            predictions, _ = self.model(cell_indices, drug_indices, genomic_features)
            return predictions.cpu().numpy()

# Custom Dataset class
class DrugResponseDataset(torch.utils.data.Dataset):
    def __init__(self, cell_indices, drug_indices, genomic_data, targets):
        self.cell_indices = cell_indices
        self.drug_indices = drug_indices
        self.genomic_data = genomic_data
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return (
            self.cell_indices[idx],
            self.drug_indices[idx],
            self.genomic_data[idx],
            self.targets[idx]
        )

# Utility functions
def create_data_loaders(train_data, val_data, batch_size=32):
    """Create PyTorch DataLoader objects"""
    train_dataset = DrugResponseDataset(*train_data)
    val_dataset = DrugResponseDataset(*val_data)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader

def train_drpo_model(model, train_data, val_data, epochs=100, lr=0.001, device='cpu'):
    """Complete training function for DRPO model"""
    trainer = DRPOTrainer(model, device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    train_loader, val_loader = create_data_loaders(train_data, val_data)
    
    best_pcc = -1
    best_model_state = None
    
    logger.info("Starting DRPO training...")
    
    for epoch in range(epochs):
        # Training
        train_loss = trainer.train_epoch(train_loader, optimizer, criterion)
        
        # Validation
        val_metrics = trainer.validate(val_loader, criterion)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['mse'])
        
        # Track best model
        if val_metrics['pcc'] > best_pcc:
            best_pcc = val_metrics['pcc']
            best_model_state = model.state_dict().copy()
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                       f"Val MSE={val_metrics['mse']:.4f}, "
                       f"Val PCC={val_metrics['pcc']:.4f}")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    logger.info(f"Training completed. Best PCC: {best_pcc:.4f}")
    return trainer, best_pcc

# Example usage
if __name__ == "__main__":
    # Example with dummy data
    n_cells, n_drugs, genomic_dim = 100, 50, 1000
    
    # Create model
    model = DRPOModel(n_cells, n_drugs, genomic_dim)
    
    # Dummy data
    cell_indices = np.random.randint(0, n_cells, 1000)
    drug_indices = np.random.randint(0, n_drugs, 1000)
    genomic_data = np.random.randn(1000, genomic_dim)
    targets = np.random.randn(1000)
    
    # Split data
    split_idx = int(0.8 * len(targets))
    train_data = (cell_indices[:split_idx], drug_indices[:split_idx], 
                  genomic_data[:split_idx], targets[:split_idx])
    val_data = (cell_indices[split_idx:], drug_indices[split_idx:], 
                genomic_data[split_idx:], targets[split_idx:])
    
    # Train model
    trainer, best_pcc = train_drpo_model(model, train_data, val_data, epochs=20)
    print(f"Best PCC achieved: {best_pcc:.4f}")
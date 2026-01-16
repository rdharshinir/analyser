"""
DeepCDR Model Implementation
Hybrid architecture with Graph Convolutional Networks and Parallel CNNs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import logging

try:
    import deepchem as dc
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("Warning: DeepChem/RDKit not available, using simplified version")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MolecularGCN(nn.Module):
    """Graph Convolutional Network for molecular structure processing"""
    
    def __init__(self, node_dim=75, hidden_dim=128, num_layers=3):
        super(MolecularGCN, self).__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Graph convolutional layers
        self.gc_layers = nn.ModuleList()
        self.gc_layers.append(nn.Linear(node_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.gc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, node_features, adjacency_matrix):
        """
        Forward pass through GCN
        node_features: [batch_size, num_atoms, node_dim]
        adjacency_matrix: [batch_size, num_atoms, num_atoms]
        """
        h = node_features
        
        for i, (gc_layer, bn) in enumerate(zip(self.gc_layers, self.batch_norms)):
            # Graph convolution: AXW
            h = torch.bmm(adjacency_matrix, h)  # AX
            h = gc_layer(h)  # (AX)W
            h = bn(h.transpose(1, 2)).transpose(1, 2)  # Batch norm
            
            if i < len(self.gc_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=0.3, training=self.training)
        
        # Global average pooling
        batch_size, num_atoms, hidden_dim = h.size()
        h = h.transpose(1, 2)  # [batch_size, hidden_dim, num_atoms]
        pooled = self.pool(h).squeeze(2)  # [batch_size, hidden_dim]
        
        return pooled

class ParallelCNN(nn.Module):
    """Parallel CNN for multi-omics data processing"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], kernel_sizes=[3, 5, 7]):
        super(ParallelCNN, self).__init__()
        
        self.input_dim = input_dim
        self.kernel_sizes = kernel_sizes
        
        # Convert 1D features to 2D for CNN processing
        self.feature_expander = nn.Linear(input_dim, input_dim * 4)
        
        # Parallel CNN branches
        self.cnn_branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(4, 32, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(16)
            )
            self.cnn_branches.append(branch)
        
        # Fusion layer
        total_features = 64 * len(kernel_sizes) * 16
        self.fusion = nn.Sequential(
            nn.Linear(total_features, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            self.hidden_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
            
        self.output_dim = prev_dim
        
    def forward(self, x):
        """
        x: [batch_size, input_dim]
        """
        # Expand features for CNN processing
        expanded = self.feature_expander(x)  # [batch_size, input_dim * 4]
        expanded = expanded.view(expanded.size(0), 4, -1)  # [batch_size, 4, seq_len]
        
        # Process through parallel CNN branches
        cnn_outputs = []
        for branch in self.cnn_branches:
            branch_out = branch(expanded)  # [batch_size, 64, 16]
            branch_out = branch_out.view(branch_out.size(0), -1)  # [batch_size, 64*16]
            cnn_outputs.append(branch_out)
        
        # Concatenate all branch outputs
        concatenated = torch.cat(cnn_outputs, dim=1)  # [batch_size, 64*3*16]
        
        # Fusion and hidden layers
        fused = self.fusion(concatenated)
        
        for layer in self.hidden_layers:
            fused = layer(fused)
            
        return fused

class DeepCDR(nn.Module):
    """Complete DeepCDR Model"""
    
    def __init__(self, genomic_dim=1000, drug_dim=75, num_atoms_max=50, 
                 gcn_hidden=128, cnn_hidden=[256, 128], final_hidden=[512, 256]):
        super(DeepCDR, self).__init__()
        
        self.genomic_dim = genomic_dim
        self.drug_dim = drug_dim
        self.num_atoms_max = num_atoms_max
        
        # Molecular processing
        self.mol_gcn = MolecularGCN(node_dim=drug_dim, hidden_dim=gcn_hidden)
        
        # Multi-omics processing (parallel CNNs)
        self.mutation_cnn = ParallelCNN(genomic_dim, cnn_hidden)
        self.expression_cnn = ParallelCNN(genomic_dim, cnn_hidden)
        self.methylation_cnn = ParallelCNN(genomic_dim, cnn_hidden)
        
        # Feature fusion
        total_features = gcn_hidden + 3 * cnn_hidden[-1]  # GCN + 3 CNNs
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, final_hidden[0]),
            nn.BatchNorm1d(final_hidden[0]),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Final prediction layers
        self.predictor = nn.ModuleList()
        prev_dim = final_hidden[0]
        for hidden_dim in final_hidden[1:]:
            self.predictor.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.4)
            ])
            prev_dim = hidden_dim
            
        self.predictor.append(nn.Linear(prev_dim, 1))
        
    def forward(self, genomic_data, mol_node_features, mol_adjacency):
        """
        Forward pass
        genomic_data: [batch_size, genomic_dim]
        mol_node_features: [batch_size, num_atoms, drug_dim]
        mol_adjacency: [batch_size, num_atoms, num_atoms]
        """
        # Process molecular data
        mol_features = self.mol_gcn(mol_node_features, mol_adjacency)
        
        # Process multi-omics data (using same data for all modalities in this implementation)
        mutation_features = self.mutation_cnn(genomic_data)
        expression_features = self.expression_cnn(genomic_data)
        methylation_features = self.methylation_cnn(genomic_data)
        
        # Fuse all features
        fused_features = torch.cat([
            mol_features,
            mutation_features,
            expression_features,
            methylation_features
        ], dim=1)
        
        # Final prediction
        x = self.fusion_layer(fused_features)
        for layer in self.predictor:
            x = layer(x)
            
        return x.squeeze()

class DeepCDRTrainer:
    """Trainer for DeepCDR model"""
    
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
            genomic_data, mol_nodes, mol_adj, targets = batch
            
            genomic_data = genomic_data.to(self.device)
            mol_nodes = mol_nodes.to(self.device)
            mol_adj = mol_adj.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            
            predictions = self.model(genomic_data, mol_nodes, mol_adj)
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
        
        with torch.no_grad():
            for batch in val_loader:
                genomic_data, mol_nodes, mol_adj, batch_targets = batch
                
                genomic_data = genomic_data.to(self.device)
                mol_nodes = mol_nodes.to(self.device)
                mol_adj = mol_adj.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                batch_predictions = self.model(genomic_data, mol_nodes, mol_adj)
                
                predictions.extend(batch_predictions.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mse = mean_squared_error(targets, predictions)
        pcc, _ = pearsonr(targets, predictions)
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'pcc': pcc
        }
        
        self.val_metrics.append(metrics)
        return metrics

# Helper functions for molecular processing
def smiles_to_graph(smiles, max_atoms=50, node_dim=75):
    """
    Convert SMILES to graph representation
    Returns node features and adjacency matrix
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Return dummy molecule
            return np.zeros((max_atoms, node_dim)), np.eye(max_atoms)
        
        # Pad or truncate atoms
        atoms = list(mol.GetAtoms())
        if len(atoms) > max_atoms:
            atoms = atoms[:max_atoms]
        
        # Node features (simplified atom features)
        node_features = np.zeros((max_atoms, node_dim))
        for i, atom in enumerate(atoms):
            # Basic atom features
            atomic_num = atom.GetAtomicNum()
            node_features[i, :min(node_dim, 10)] = [
                atomic_num,
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                atom.GetImplicitValence(),
                int(atom.GetIsAromatic()),
                atom.GetMass(),
                0, 0, 0  # Padding
            ][:min(node_dim, 10)]
        
        # Adjacency matrix
        adj_matrix = np.zeros((max_atoms, max_atoms))
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            if begin_idx < max_atoms and end_idx < max_atoms:
                adj_matrix[begin_idx, end_idx] = 1
                adj_matrix[end_idx, begin_idx] = 1
        
        # Add self-loops
        np.fill_diagonal(adj_matrix, 1)
        
        return node_features.astype(np.float32), adj_matrix.astype(np.float32)
        
    except Exception as e:
        logger.warning(f"Error processing SMILES {smiles}: {e}")
        return np.zeros((max_atoms, node_dim)), np.eye(max_atoms)

# Dataset class
class DeepCDRDataset(torch.utils.data.Dataset):
    def __init__(self, genomic_data, smiles_list, targets, max_atoms=50):
        self.genomic_data = genomic_data
        self.smiles_list = smiles_list
        self.targets = targets
        self.max_atoms = max_atoms
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        genomic = self.genomic_data[idx]
        smiles = self.smiles_list[idx] if idx < len(self.smiles_list) else "CCO"  # Ethanol as fallback
        target = self.targets[idx]
        
        # Convert SMILES to graph
        node_features, adj_matrix = smiles_to_graph(smiles, self.max_atoms)
        
        return (
            torch.FloatTensor(genomic),
            torch.FloatTensor(node_features),
            torch.FloatTensor(adj_matrix),
            torch.FloatTensor([target])
        )

def create_deeppcdr_loaders(train_data, val_data, batch_size=32):
    """Create data loaders for DeepCDR"""
    train_dataset = DeepCDRDataset(*train_data)
    val_dataset = DeepCDRDataset(*val_data)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader

def train_deeppcdr_model(model, train_data, val_data, epochs=100, lr=0.001, device='cpu'):
    """Complete training function for DeepCDR"""
    trainer = DeepCDRTrainer(model, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    train_loader, val_loader = create_deeppcdr_loaders(train_data, val_data)
    
    best_pcc = -1
    best_model_state = None
    
    logger.info("Starting DeepCDR training...")
    
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
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    logger.info(f"DeepCDR training completed. Best PCC: {best_pcc:.4f}")
    return trainer, best_pcc

# Example usage
if __name__ == "__main__":
    # Test with dummy data
    genomic_dim, drug_dim, max_atoms = 1000, 75, 50
    batch_size = 16
    
    # Create model
    model = DeepCDR(genomic_dim=genomic_dim, drug_dim=drug_dim, num_atoms_max=max_atoms)
    
    # Dummy data
    genomic_data = np.random.randn(100, genomic_dim).astype(np.float32)
    smiles_list = ["CCO", "CCN", "CCC"] * 33 + ["CCO"]  # 100 SMILES
    targets = np.random.randn(100).astype(np.float32)
    
    # Split data
    split_idx = 80
    train_data = (genomic_data[:split_idx], smiles_list[:split_idx], targets[:split_idx])
    val_data = (genomic_data[split_idx:], smiles_list[split_idx:], targets[split_idx:])
    
    # Train model
    trainer, best_pcc = train_deeppcdr_model(model, train_data, val_data, epochs=20)
    print(f"DeepCDR Best PCC: {best_pcc:.4f}")
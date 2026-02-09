"""
Graph Neural Network Training for Bridge Criticality Prediction
Oklahoma Bridge Network Analysis

This script trains a GCN model to predict bridge criticality using:
- Individual bridge features (150-200 features)
- Spatial relationships between bridges (NEAR edges)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.serialization
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr
from torch_geometric.nn import GATConv, GCNConv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import json
import time
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("BRIDGE CRITICALITY PREDICTION - GNN TRAINING")
print("="*70)

# ==================== CONFIGURATION ====================
CONFIG = {
    'model_type': 'GCN',  # Options: 'GCN' or 'GAT'
    'hidden_dim': 128,
    'num_layers': 3,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'weight_decay': 5e-4,
    'num_epochs': 200,
    'patience': 20,  # Early stopping patience
    'class_weight': True,  # Handle class imbalance
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"\n🖥️  Using device: {CONFIG['device']}")
if CONFIG['device'] == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ==================== LOAD DATA ====================
print("\n📦 Loading graph data...")
data_path = DATA_DIR / "comprehensive_bridge_graph.pt"
with torch.serialization.safe_globals([DataEdgeAttr]):
    data = torch.load(data_path, weights_only=False)

print(f"\n📊 Dataset Statistics:")
print(f"   Nodes (bridges): {data.num_nodes:,}")
print(f"   Edges (NEAR relationships): {data.num_edges:,}")
print(f"   Node features: {data.num_node_features}")
print(f"   Classes: {data.num_classes}")
print(f"   Average degree: {data.num_edges / data.num_nodes:.2f}")

print(f"\n📋 Data Split:")
print(f"   Training: {data.train_mask.sum():,} bridges")
print(f"   Validation: {data.val_mask.sum():,} bridges")
print(f"   Test: {data.test_mask.sum():,} bridges")

# Check label distribution
train_labels = data.y[data.train_mask]
val_labels = data.y[data.val_mask]
test_labels = data.y[data.test_mask]

print(f"\n🎯 Label Distribution:")
print(f"   Training - Critical: {train_labels.sum()}/{len(train_labels)} ({train_labels.sum()/len(train_labels)*100:.1f}%)")
print(f"   Validation - Critical: {val_labels.sum()}/{len(val_labels)} ({val_labels.sum()/len(val_labels)*100:.1f}%)")
print(f"   Test - Critical: {test_labels.sum()}/{len(test_labels)} ({test_labels.sum()/len(test_labels)*100:.1f}%)")

# Move data to device
data = data.to(CONFIG['device'])

# ==================== MODEL DEFINITION ====================

class BridgeGCN(torch.nn.Module):
    """
    Graph Convolutional Network for Bridge Criticality Prediction
    
    Architecture:
    - Input: Node features (150-200 dimensions)
    - Hidden: 3 GCN layers with ReLU and Dropout
    - Output: 2 classes (Critical, Good)
    """
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.3):
        super(BridgeGCN, self).__init__()
        
        # Graph Convolutional Layers
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, hidden_dim // 4)
        
        # Final classifier
        self.classifier = torch.nn.Linear(hidden_dim // 4, num_classes)
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classification
        x = self.classifier(x)
        
        return x

class BridgeGAT(torch.nn.Module):
    """
    Graph Attention Network for Bridge Criticality Prediction
    
    Uses attention mechanism to weigh neighbor importance
    """
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.3, heads=4):
        super(BridgeGAT, self).__init__()
        
        # Graph Attention Layers
        self.conv1 = GATConv(num_features, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim // 2, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim // 2 * heads, hidden_dim // 4, heads=1, dropout=dropout)
        
        # Final classifier
        self.classifier = torch.nn.Linear(hidden_dim // 4, num_classes)
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.classifier(x)
        
        return x

# ==================== INITIALIZE MODEL ====================
print(f"\n🏗️  Building {CONFIG['model_type']} model...")

if CONFIG['model_type'] == 'GCN':
    model = BridgeGCN(
        num_features=data.num_node_features,
        hidden_dim=CONFIG['hidden_dim'],
        num_classes=data.num_classes,
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
else:
    model = BridgeGAT(
        num_features=data.num_node_features,
        hidden_dim=CONFIG['hidden_dim'],
        num_classes=data.num_classes,
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])

print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ==================== LOSS FUNCTION WITH CLASS WEIGHTS ====================
if CONFIG['class_weight']:
    # Calculate class weights to handle imbalance
    n_critical = train_labels.sum().item()
    n_good = len(train_labels) - n_critical
    
    # Weight inversely proportional to class frequency
    weight_critical = len(train_labels) / (2 * n_critical)
    weight_good = len(train_labels) / (2 * n_good)
    
    class_weights = torch.tensor([weight_good, weight_critical]).to(CONFIG['device'])
    print(f"\n⚖️  Class weights: Good={weight_good:.3f}, Critical={weight_critical:.3f}")
else:
    class_weights = None

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# ==================== OPTIMIZER ====================
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay']
)

# ==================== TRAINING FUNCTIONS ====================

def train():
    """Single training epoch"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x, data.edge_index)
    
    # Calculate loss only on training nodes
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def evaluate(mask):
    """Evaluate model on given mask (train/val/test)"""
    model.eval()
    
    # Forward pass
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    # Get predictions and labels for this mask
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()
    y_prob = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()  # Probability of class 1
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC-ROC (if both classes present)
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = 0.0
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

# ==================== TRAINING LOOP ====================
print("\n" + "="*70)
print("TRAINING STARTED")
print("="*70)

# Track metrics
history = {
    'train_loss': [],
    'val_accuracy': [],
    'val_f1': [],
    'val_auc': []
}

best_val_f1 = 0
best_epoch = 0
patience_counter = 0

start_time = time.time()

for epoch in range(1, CONFIG['num_epochs'] + 1):
    # Train
    loss = train()
    history['train_loss'].append(loss)
    
    # Evaluate
    train_metrics = evaluate(data.train_mask)
    val_metrics = evaluate(data.val_mask)
    
    history['val_accuracy'].append(val_metrics['accuracy'])
    history['val_f1'].append(val_metrics['f1'])
    history['val_auc'].append(val_metrics['auc'])
    
    # Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | "
              f"Loss: {loss:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f}")
    
    # Early stopping based on F1 score
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        best_epoch = epoch
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': best_val_f1,
            'config': CONFIG
        }, MODELS_DIR / 'best_bridge_gnn_model.pt')
    else:
        patience_counter += 1
    
    # Stop if no improvement
    if patience_counter >= CONFIG['patience']:
        print(f"\n⏹️  Early stopping at epoch {epoch}")
        print(f"   Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
        break

training_time = time.time() - start_time
print(f"\n✅ Training completed in {training_time:.2f} seconds")
print(f"   Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")

# ==================== LOAD BEST MODEL & FINAL EVALUATION ====================
print("\n" + "="*70)
print("FINAL EVALUATION ON TEST SET")
print("="*70)

# Load best model
checkpoint = torch.load(MODELS_DIR / 'best_bridge_gnn_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate on all sets
train_results = evaluate(data.train_mask)
val_results = evaluate(data.val_mask)
test_results = evaluate(data.test_mask)

print("\n📊 Final Metrics:")
print(f"\n{'Metric':<15} {'Train':<10} {'Validation':<12} {'Test':<10}")
print("-" * 50)
print(f"{'Accuracy':<15} {train_results['accuracy']:<10.4f} {val_results['accuracy']:<12.4f} {test_results['accuracy']:<10.4f}")
print(f"{'Precision':<15} {train_results['precision']:<10.4f} {val_results['precision']:<12.4f} {test_results['precision']:<10.4f}")
print(f"{'Recall':<15} {train_results['recall']:<10.4f} {val_results['recall']:<12.4f} {test_results['recall']:<10.4f}")
print(f"{'F1-Score':<15} {train_results['f1']:<10.4f} {val_results['f1']:<12.4f} {test_results['f1']:<10.4f}")
print(f"{'AUC-ROC':<15} {train_results['auc']:<10.4f} {val_results['auc']:<12.4f} {test_results['auc']:<10.4f}")

# Confusion Matrix
print("\n📊 Test Set Confusion Matrix:")
cm = confusion_matrix(test_results['y_true'], test_results['y_pred'])
print(f"\n                 Predicted")
print(f"                 Good  Critical")
print(f"Actual  Good     {cm[0,0]:<6} {cm[0,1]:<6}")
print(f"        Critical {cm[1,0]:<6} {cm[1,1]:<6}")

# Detailed Classification Report
print("\n📋 Detailed Classification Report (Test Set):")
print(classification_report(
    test_results['y_true'], 
    test_results['y_pred'],
    target_names=['Good', 'Critical'],
    digits=4
))

# ==================== SAVE RESULTS ====================
print("\n💾 Saving results...")

# Save metrics
results = {
    'config': CONFIG,
    'training_time': training_time,
    'best_epoch': best_epoch,
    'train_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v.tolist() 
                     for k, v in train_results.items() if k not in ['y_true', 'y_pred', 'y_prob']},
    'val_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v.tolist() 
                   for k, v in val_results.items() if k not in ['y_true', 'y_pred', 'y_prob']},
    'test_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v.tolist() 
                    for k, v in test_results.items() if k not in ['y_true', 'y_pred', 'y_prob']},
    'confusion_matrix': cm.tolist(),
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

results_json = RESULTS_DIR / "training_results.json"
with open(results_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"   ✓ Saved {results_json}")

# ==================== VISUALIZATIONS ====================
print("\n📈 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Training Loss
axes[0, 0].plot(history['train_loss'], label='Training Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss Over Time')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Validation Metrics
axes[0, 1].plot(history['val_accuracy'], label='Accuracy', linewidth=2)
axes[0, 1].plot(history['val_f1'], label='F1-Score', linewidth=2)
axes[0, 1].plot(history['val_auc'], label='AUC-ROC', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_title('Validation Metrics Over Time')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label='Best Epoch')

# 3. Confusion Matrix Heatmap
im = axes[1, 0].imshow(cm, cmap='Blues', aspect='auto')
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(['Good', 'Critical'])
axes[1, 0].set_yticklabels(['Good', 'Critical'])
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_title('Confusion Matrix (Test Set)')

# Add text annotations
for i in range(2):
    for j in range(2):
        text = axes[1, 0].text(j, i, str(cm[i, j]),
                              ha="center", va="center", color="black", fontsize=20)

plt.colorbar(im, ax=axes[1, 0])

# 4. ROC Curve
fpr, tpr, _ = roc_curve(test_results['y_true'], test_results['y_prob'])
axes[1, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {test_results["auc"]:.3f})')
axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curve (Test Set)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
results_plot = RESULTS_DIR / "training_results.png"
plt.savefig(results_plot, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved {results_plot}")

print("\n" + "="*70)
print("✅ TRAINING AND EVALUATION COMPLETE!")
print("="*70)
print("\n📁 Generated Files:")
print(f"   - {MODELS_DIR / 'best_bridge_gnn_model.pt'} (trained model)")
print(f"   - {results_json} (metrics)")
print(f"   - {results_plot} (visualizations)")
print("\n🎯 Next Steps:")
print("   1. Analyze feature importance")
print("   2. Generate predictions for all bridges")
print("   3. Create risk map visualization")
print("   4. Perform error analysis")
print("="*70)

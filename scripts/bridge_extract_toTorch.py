"""
Convert comprehensive bridge graph data to PyTorch Geometric format
Use this on your machine with working PyTorch
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("CONVERTING COMPREHENSIVE GRAPH DATA TO PYTORCH")
print("="*70)

try:
    # Load the comprehensive data
    source_path = DATA_DIR / "comprehensive_bridge_graph.pkl"
    print(f"\n📦 Loading {source_path}...")
    with open(source_path, 'rb') as f:
        graph_data = pickle.load(f)
    
    print("✓ Data loaded successfully")
    
    # Show what we got
    print(f"\n📊 Data Summary:")
    print(f"  Bridges: {graph_data['num_nodes']:,}")
    print(f"  Edges: {graph_data['num_edges']:,}")
    print(f"  Features: {graph_data['num_features']:,}")
    print(f"  Target: {graph_data['target_column']}")
    
    # Convert to PyTorch tensors
    print("\n🔄 Converting to PyTorch tensors...")
    
    X_tensor = torch.FloatTensor(graph_data['features'])
    print(f"  ✓ Features: {X_tensor.shape}")
    
    # Convert edge_list to edge_index format (2 x num_edges)
    edge_index = torch.LongTensor(graph_data['edge_list']).t().contiguous()
    print(f"  ✓ Edge index: {edge_index.shape}")
    
    y = torch.LongTensor(graph_data['labels'])
    print(f"  ✓ Labels: {y.shape}")
    
    # Create masks
    n = graph_data['num_nodes']
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    
    train_mask[graph_data['train_indices']] = True
    val_mask[graph_data['val_indices']] = True
    test_mask[graph_data['test_indices']] = True
    
    print(f"  ✓ Train/Val/Test masks: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")
    
    # Create PyG Data object
    print("\n📊 Creating PyTorch Geometric Data object...")
    
    data = Data(
        x=X_tensor,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    # Add comprehensive metadata
    data.bridge_ids = graph_data['bridge_ids']
    data.feature_names = graph_data['feature_names']
    data.num_classes = 2
    data.target_column = graph_data['target_column']
    data.id_to_idx = graph_data['id_to_idx']
    
    print("✓ PyG Data object created with all metadata")
    
    # Save
    print("\n💾 Saving PyTorch format...")
    target_path = DATA_DIR / "comprehensive_bridge_graph.pt"
    torch.save(data, target_path)
    
    # Also save a separate metadata file
    metadata = {
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'num_features': data.num_node_features,
        'num_classes': data.num_classes,
        'target_column': data.target_column,
        'train_size': int(train_mask.sum()),
        'val_size': int(val_mask.sum()),
        'test_size': int(test_mask.sum())
    }
    
    metadata_path = DATA_DIR / "pytorch_graph_info.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ CONVERSION SUCCESSFUL!")
    print("="*70)
    
    print("\n📁 Created Files:")
    print(f"  - {target_path} (PyTorch Geometric Data)")
    print(f"  - {metadata_path} (metadata)")
    
    print("\n📊 PyTorch Graph Statistics:")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.num_edges:,}")
    print(f"  Features: {data.num_node_features:,}")
    print(f"  Classes: {data.num_classes}")
    print(f"  Target: {data.target_column}")
    
    print("\n✅ Ready for GNN training!")
    
    print("\n💡 Quick Start:")
    print("  import torch")
    print("  from torch_geometric.data import Data")
    print(f"  data = torch.load('{target_path}')")
    print("  print(f'Loaded {data.num_nodes} bridges with {data.num_node_features} features')")
    
except FileNotFoundError:
    print("\n❌ ERROR: comprehensive_bridge_graph.pkl not found!")
    print("   Make sure you ran comprehensive_bridge_extract.py first")
    print(f"   and transferred the .pkl file to {DATA_DIR}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

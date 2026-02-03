import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# --- NEO4J CONFIGURATION ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "MikalBridges"

print("="*70)
print("COMPREHENSIVE BRIDGE GRAPH DATA EXTRACTION - ALL FEATURES")
print("="*70)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_comprehensive_bridge_data():
    """
    Fetch EVERY property from Bridge nodes + all relationship data
    No shortcuts, no filtering - get everything
    """
    query = """
    MATCH (b:Bridge)
    OPTIONAL MATCH (b)-[e:EXPOSED_TO]->(cz:ClimateZone)
    OPTIONAL MATCH (b)-[t:HAS_TRAFFIC]->(tp:TrafficPattern)
    RETURN 
        b {.*} as bridge_props,
        cz {.*} as climate_props,
        tp {.*} as traffic_props
    """
    
    print("\n📊 Fetching ALL bridge data from Neo4j...")
    print("   This may take a moment for large datasets...")
    
    with driver.session() as session:
        result = session.run(query)
        records = []
        
        for record in result:
            # Start with all bridge properties
            row = dict(record['bridge_props']) if record['bridge_props'] else {}
            
            # Add climate properties with prefix
            if record['climate_props']:
                for key, val in record['climate_props'].items():
                    row[f'climate_{key}'] = val
            
            # Add traffic properties with prefix
            if record['traffic_props']:
                for key, val in record['traffic_props'].items():
                    row[f'traffic_{key}'] = val
            
            records.append(row)
        
    df = pd.DataFrame(records)
    print(f"✓ Loaded {len(df)} bridges")
    print(f"✓ Total columns extracted: {len(df.columns)}")
    
    return df

def identify_column_types(df):
    """
    Intelligently identify what type each column is
    WITHOUT removing any columns
    """
    print("\n🔍 Analyzing all columns...")
    
    # These are the ONLY columns we exclude from features (structural identifiers only)
    ID_COLS = [
        'STRUCTURE_NUMBER', 'structureNumber', 'structure_number',
        'NBI_8 - Structure Number', 'LATEST_STRUCTURE_NUMBER',
        'NBI_2025 NBI Structure Number'
    ]
    
    # Target variables (condition ratings we might predict)
    TARGET_KEYWORDS = [
        'Condition Rating', 'Condition Code', 'Bridge Condition',
        'currentDeckCR', 'currentSuperCR', 'currentSubCR',
        'NBI_58', 'NBI_59', 'NBI_60', 'NBI_61', 'NBI_62'
    ]
    
    id_cols = []
    target_cols = []
    numeric_cols = []
    categorical_cols = []
    
    for col in df.columns:
        # Check if ID column
        if col in ID_COLS:
            id_cols.append(col)
            continue
        
        # Check if target column
        if any(kw in col for kw in TARGET_KEYWORDS):
            target_cols.append(col)
            continue
        
        # Analyze the column data
        dtype = df[col].dtype
        non_null = df[col].dropna()
        
        if len(non_null) == 0:
            # Empty column, skip
            continue
        
        # Try to determine if numeric or categorical
        if dtype in ['int64', 'float64', 'int32', 'float32']:
            # Already numeric
            numeric_cols.append(col)
        elif dtype == 'object':
            # Check if it can be converted to numeric
            try:
                pd.to_numeric(non_null.head(100), errors='raise')
                # Successfully converted sample - likely numeric
                numeric_cols.append(col)
            except:
                # Contains non-numeric values
                n_unique = non_null.nunique()
                total = len(non_null)
                
                # If unique ratio is low, it's categorical
                # If unique ratio is high, might be text/codes - still categorical
                if n_unique <= 100 or (n_unique / total) < 0.5:
                    categorical_cols.append(col)
                else:
                    # High cardinality - could be codes, treat as categorical
                    categorical_cols.append(col)
        else:
            # Boolean, datetime, etc - convert to numeric if possible
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_cols.append(col)
            except:
                categorical_cols.append(col)
    
    print(f"\n📋 Column Classification:")
    print(f"  🆔 ID Columns: {len(id_cols)}")
    print(f"  🎯 Target Columns: {len(target_cols)}")
    print(f"  🔢 Numeric Columns: {len(numeric_cols)}")
    print(f"  📊 Categorical Columns: {len(categorical_cols)}")
    print(f"  📦 TOTAL FEATURE COLUMNS: {len(numeric_cols) + len(categorical_cols)}")
    
    return {
        'id': id_cols,
        'target': target_cols,
        'numeric': numeric_cols,
        'categorical': categorical_cols
    }

def process_all_features(df, column_types):
    """
    Process ALL features comprehensively
    """
    print("\n⚙️  Processing ALL features...")
    
    numeric_cols = column_types['numeric']
    categorical_cols = column_types['categorical']
    
    all_features = []
    feature_names = []
    
    # ==================== NUMERIC FEATURES ====================
    if numeric_cols:
        print(f"\n  Processing {len(numeric_cols)} numeric features...")
        
        numeric_df = df[numeric_cols].copy()
        
        # Convert all to numeric, coercing errors
        for col in numeric_cols:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
        
        # Check for completely empty columns
        valid_numeric_cols = []
        for col in numeric_cols:
            if numeric_df[col].notna().sum() > 0:
                valid_numeric_cols.append(col)
            else:
                print(f"    ⚠️  Skipping '{col}' - all values missing")
        
        if valid_numeric_cols:
            numeric_df = numeric_df[valid_numeric_cols]
            
            # Impute missing values with median (robust to outliers)
            imputer = SimpleImputer(strategy='median')
            numeric_imputed = imputer.fit_transform(numeric_df)
            
            # Standardize (mean=0, std=1)
            scaler = StandardScaler()
            numeric_scaled = scaler.fit_transform(numeric_imputed)
            
            all_features.append(numeric_scaled)
            feature_names.extend([f"num_{col}" for col in valid_numeric_cols])
            
            print(f"    ✓ Processed {len(valid_numeric_cols)} numeric features")
    
    # ==================== CATEGORICAL FEATURES ====================
    if categorical_cols:
        print(f"\n  Processing {len(categorical_cols)} categorical features...")
        
        categorical_df = df[categorical_cols].copy()
        
        # Convert to string and handle missing
        for col in categorical_cols:
            categorical_df[col] = categorical_df[col].astype(str).fillna('MISSING').replace('nan', 'MISSING')
        
        # One-hot encode each categorical
        encoded_features = []
        encoded_names = []
        
        for col in categorical_cols:
            unique_vals = categorical_df[col].unique()
            
            # Skip if only one unique value (no information)
            if len(unique_vals) <= 1:
                continue
            
            # For high-cardinality categoricals (>50 unique), use label encoding
            if len(unique_vals) > 50:
                # Label encode (convert to integers 0, 1, 2, ...)
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                encoded = le.fit_transform(categorical_df[col]).reshape(-1, 1)
                encoded_features.append(encoded)
                encoded_names.append(f"cat_encoded_{col}")
            else:
                # One-hot encode (create binary columns for each category)
                dummies = pd.get_dummies(categorical_df[col], prefix=f"cat_{col}", drop_first=False)
                encoded_features.append(dummies.values)
                encoded_names.extend(dummies.columns.tolist())
        
        if encoded_features:
            categorical_encoded = np.hstack(encoded_features)
            all_features.append(categorical_encoded)
            feature_names.extend(encoded_names)
            
            print(f"    ✓ Processed {len(categorical_cols)} categorical features")
            print(f"    ✓ Created {categorical_encoded.shape[1]} categorical feature columns")
    
    # ==================== COMBINE ALL FEATURES ====================
    if not all_features:
        raise ValueError("No valid features extracted!")
    
    X = np.hstack(all_features).astype(np.float32)
    
    print(f"\n  ✅ TOTAL FEATURE MATRIX: {X.shape}")
    print(f"     {X.shape[0]} bridges × {X.shape[1]} features")
    
    return X, feature_names

def extract_labels(df, column_types):
    """
    Extract target labels for prediction
    """
    print("\n🎯 Extracting labels...")
    
    target_cols = column_types['target']
    
    # Priority targets for prediction
    priority_order = [
        'NBI_58 - Deck Condition Rating',
        'currentDeckCR',
        'NBI_CAT10 - Bridge Condition',
        'NBI_59 - Superstructure Condition Rating',
        'NBI_60 - Substructure Condition Rating'
    ]
    
    target_col = None
    for priority in priority_order:
        if priority in target_cols:
            target_col = priority
            break
    
    # If no priority match, use first available target
    if target_col is None and target_cols:
        target_col = target_cols[0]
    
    if target_col:
        print(f"  Using '{target_col}' as prediction target")
        
        # Convert to numeric
        y_raw = pd.to_numeric(df[target_col], errors='coerce')
        
        # Create binary classification: Critical (≤5) vs Good (>5)
        # Using threshold of 5 based on NBI rating scale (0-9, where ≤4 is poor, 5 is fair)
        y = (y_raw <= 5).astype(int).values
        
        # Handle NaN values (set to majority class)
        nan_mask = pd.isna(y_raw)
        if nan_mask.any():
            majority_class = 1 if y.sum() > len(y)/2 else 0
            y[nan_mask] = majority_class
            print(f"  ⚠️  Filled {nan_mask.sum()} missing labels with majority class ({majority_class})")
        
        n_critical = y.sum()
        n_good = len(y) - n_critical
        print(f"\n  Label Distribution:")
        print(f"    Critical (≤5): {n_critical} ({n_critical/len(y)*100:.1f}%)")
        print(f"    Good (>5): {n_good} ({n_good/len(y)*100:.1f}%)")
    else:
        print("  ⚠️  No target column found - creating dummy labels")
        y = np.zeros(len(df), dtype=int)
        target_col = "dummy_target"
    
    return y, target_col

def extract_edges(bridge_ids):
    """
    Extract NEAR relationship edges
    """
    query = """
    MATCH (b1:Bridge)-[:NEAR]->(b2:Bridge)
    RETURN b1.structureNumber as source, b2.structureNumber as target
    """
    
    print("\n🔗 Extracting graph edges...")
    with driver.session() as session:
        result = session.run(query)
        edges = [(str(record['source']), str(record['target'])) for record in result]
    
    print(f"  Found {len(edges)} NEAR relationships")
    
    # Create ID to index mapping
    id_to_idx = {str(bid): i for i, bid in enumerate(bridge_ids)}
    
    # Convert to index pairs
    edge_list = []
    for src, tgt in edges:
        if src in id_to_idx and tgt in id_to_idx:
            edge_list.append([id_to_idx[src], id_to_idx[tgt]])
    
    if not edge_list:
        print("  ⚠️  No valid edges found - creating self-loops")
        edge_list = [[i, i] for i in range(len(bridge_ids))]
    
    print(f"  ✓ Created {len(edge_list)} valid edges")
    
    return edge_list, id_to_idx

def main():
    try:
        # 1. Get ALL data
        df = get_comprehensive_bridge_data()
        
        # 2. Get bridge IDs (keep them separate from features)
        id_candidates = [
            'STRUCTURE_NUMBER', 'structureNumber', 'structure_number',
            'NBI_8 - Structure Number'
        ]
        
        bridge_id_col = None
        for candidate in id_candidates:
            if candidate in df.columns:
                bridge_id_col = candidate
                break
        
        if bridge_id_col:
            bridge_ids = df[bridge_id_col].astype(str).tolist()
            print(f"\n🆔 Using '{bridge_id_col}' as bridge identifier")
        else:
            print("\n⚠️  No structure number column found - using row index")
            bridge_ids = [f"bridge_{i}" for i in range(len(df))]
        
        # 3. Classify ALL columns
        column_types = identify_column_types(df)
        
        # Show samples
        print("\n📝 Sample columns by type:")
        for cat_name, cols in column_types.items():
            if cols and cat_name != 'id':
                sample = cols[:5]
                more = f" ... and {len(cols)-5} more" if len(cols) > 5 else ""
                print(f"  {cat_name.upper()}: {sample}{more}")
        
        # 4. Process ALL features
        X, feature_names = process_all_features(df, column_types)
        
        # 5. Extract labels
        y, target_col = extract_labels(df, column_types)
        
        # 6. Get graph structure
        edge_list, id_to_idx = extract_edges(bridge_ids)
        
        # 7. Create train/val/test split
        num_nodes = len(df)
        indices = np.random.RandomState(42).permutation(num_nodes)
        
        n_train = int(0.7 * num_nodes)
        n_val = int(0.15 * num_nodes)
        
        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train:n_train+n_val].tolist()
        test_idx = indices[n_train+n_val:].tolist()
        
        # 8. Package EVERYTHING
        graph_data = {
            # Core data
            'features': X,
            'labels': y,
            'edge_list': edge_list,
            
            # Identifiers and metadata
            'bridge_ids': bridge_ids,
            'id_to_idx': id_to_idx,
            'feature_names': feature_names,
            'target_column': target_col,
            
            # Splits
            'train_indices': train_idx,
            'val_indices': val_idx,
            'test_indices': test_idx,
            
            # Stats
            'num_nodes': num_nodes,
            'num_edges': len(edge_list),
            'num_features': X.shape[1],
            
            # Column classifications (for reference)
            'column_types': column_types,
            
            # Raw data preservation
            'all_column_names': list(df.columns)
        }
        
        # 9. Save everything
        print("\n💾 Saving data...")
        
        with open('comprehensive_bridge_graph.pkl', 'wb') as f:
            pickle.dump(graph_data, f)
        print("  ✓ Saved comprehensive_bridge_graph.pkl")
        
        # Save feature mapping
        feature_df = pd.DataFrame({
            'index': range(len(feature_names)),
            'feature_name': feature_names
        })
        feature_df.to_csv('all_feature_names.csv', index=False)
        print("  ✓ Saved all_feature_names.csv")
        
        # Save metadata
        metadata = {
            'num_nodes': num_nodes,
            'num_edges': len(edge_list),
            'num_features': X.shape[1],
            'target_column': target_col,
            'num_train': len(train_idx),
            'num_val': len(val_idx),
            'num_test': len(test_idx),
            'num_numeric_features': len(column_types['numeric']),
            'num_categorical_features': len(column_types['categorical']),
            'label_distribution': {
                'critical': int(y.sum()),
                'good': int(len(y) - y.sum())
            }
        }
        
        with open('comprehensive_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("  ✓ Saved comprehensive_metadata.json")
        
        # Save original column list
        with open('all_columns_extracted.txt', 'w') as f:
            f.write("ALL COLUMNS EXTRACTED FROM NEO4J:\n")
            f.write("="*70 + "\n\n")
            for i, col in enumerate(df.columns, 1):
                f.write(f"{i}. {col}\n")
        print("  ✓ Saved all_columns_extracted.txt")
        
        # 10. Print summary
        print("\n" + "="*70)
        print("✅ COMPREHENSIVE EXTRACTION COMPLETE")
        print("="*70)
        
        print("\n📊 Final Statistics:")
        print(f"  Total Bridges: {num_nodes:,}")
        print(f"  Total Edges: {len(edge_list):,}")
        print(f"  Total Features: {X.shape[1]:,}")
        print(f"    - Numeric: {len(column_types['numeric'])}")
        print(f"    - Categorical: {len(column_types['categorical'])}")
        print(f"  Target: {target_col}")
        print(f"  Average Degree: {len(edge_list)/num_nodes:.2f}")
        
        print("\n📁 Files Created:")
        print("  - comprehensive_bridge_graph.pkl (main data)")
        print("  - all_feature_names.csv (feature mapping)")
        print("  - comprehensive_metadata.json (statistics)")
        print("  - all_columns_extracted.txt (full column list)")
        
        print("\n" + "="*70)
        print("Transfer 'comprehensive_bridge_graph.pkl' to your PyTorch machine!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.close()
        print("\n🔌 Neo4j connection closed.")

if __name__ == "__main__":
    main()
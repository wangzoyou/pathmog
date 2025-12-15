#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for loading processed heterogeneous graph data
"""

import os
import sys
import torch
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage


def add_safe_globals():
    """Add PyG storage classes to safe globals"""
    try:
        torch.serialization.add_safe_globals([BaseStorage, NodeStorage, EdgeStorage])
        print("Added PyG storage classes to safe globals")
    except Exception as e:
        print(f"Failed to add safe globals: {e}")


def safe_load_hetero_data(file_path):
    """
    Safely load heterogeneous graph data, compatible with different versions of PyTorch and PyG
    
    Parameters:
        file_path: Path to the heterogeneous graph data file
    
    Returns:
        Loaded heterogeneous graph data object, or None if loading fails
    """
    # First add safe globals
    add_safe_globals()
    
    try:
        # Method 1: Load with weights_only=False parameter (PyTorch 2.1+)
        try:
            print(f"Attempting to load with weights_only=False: {file_path}")
            data = torch.load(file_path, weights_only=False)
            print(f"Successfully loaded heterogeneous graph: {file_path}")
            return data
        except (TypeError, AttributeError) as e:
            print(f"Failed to load with weights_only=False: {e}")
        
        # Method 2: Load directly (compatible with older versions)
        try:
            print(f"Attempting direct loading: {file_path}")
            data = torch.load(file_path)
            print(f"Successfully loaded heterogeneous graph: {file_path}")
            return data
        except Exception as e:
            print(f"Failed direct loading: {e}")
        
        # Method 3: Load with map_location parameter (avoid device incompatibility)
        try:
            print(f"Attempting to load with map_location='cpu': {file_path}")
            data = torch.load(file_path, map_location='cpu')
            print(f"Successfully loaded heterogeneous graph: {file_path}")
            return data
        except Exception as e:
            print(f"Failed to load with map_location='cpu': {e}")
        
        # Method 4: Load with pickle (last attempt)
        try:
            import pickle
            print(f"Attempting to load with pickle: {file_path}")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Successfully loaded heterogeneous graph: {file_path}")
            return data
        except Exception as e:
            print(f"Failed to load with pickle: {e}")
            
        print(f"All loading methods failed")
        return None
    except Exception as e:
        print(f"Failed to load heterogeneous graph: {e}")
        return None


def check_patient_id_mapping(hetero_data):
    """
    Check patient ID mapping in the heterogeneous graph
    
    Parameters:
        hetero_data: Heterogeneous graph data object
    
    Returns:
        None
    """
    if not hasattr(hetero_data, 'patient_id_map'):
        print("Error: No patient ID mapping found in heterogeneous graph")
        return
    
    # Get patient ID mapping
    patient_id_to_idx = hetero_data.patient_id_map
    
    # Print mapping information
    print(f"Found {len(patient_id_to_idx)} entries in patient ID mapping")
    
    # Print some samples
    print("\nPatient ID mapping samples:")
    sample_count = min(10, len(patient_id_to_idx))
    sample_items = list(patient_id_to_idx.items())[:sample_count]
    for patient_id, idx in sample_items:
        print(f"Patient ID: {patient_id} -> Index: {idx}")
    
    # Check ID format
    has_tcga_prefix = any(pid.startswith("TCGA-") for pid in patient_id_to_idx.keys())
    print(f"\nID format check:")
    print(f"- Contains 'TCGA-' prefix: {'Yes' if has_tcga_prefix else 'No'}")
    
    # Print some statistics
    id_lengths = [len(pid) for pid in patient_id_to_idx.keys()]
    print(f"- ID length range: {min(id_lengths)} to {max(id_lengths)}")
    print(f"- Most common ID length: {max(set(id_lengths), key=id_lengths.count)}")
    
    # Check index range
    indices = list(patient_id_to_idx.values())
    print(f"- Index range: {min(indices)} to {max(indices)}")
    
    # Check if indices are continuous
    is_continuous = (max(indices) - min(indices) + 1) == len(indices)
    print(f"- Are indices continuous: {'Yes' if is_continuous else 'No'}")
    
    return patient_id_to_idx 
"""
Basic analysis functions for model architecture analysis
"""

import pandas as pd
from collections import defaultdict

def analyze_layer_types(task_dfs):
    """Analyze layer type distributions for each task"""
    layer_type_counts = {}
    
    for task, df in task_dfs.items():
        counts = df['Layer Type'].value_counts()
        layer_type_counts[task] = counts
    
    return layer_type_counts

def analyze_blocks(task_dfs):
    """Analyze block distributions for each task"""
    block_counts = {}
    
    for task, df in task_dfs.items():
        counts = df['Block'].value_counts()
        block_counts[task] = counts
    
    return block_counts

def common_unique_layers(task_dfs):
    """Find common and unique layer types across tasks"""
    all_layer_types = defaultdict(set)
    
    for task, df in task_dfs.items():
        all_layer_types[task] = set(df['Layer Type'].unique())
    
    # Find common layer types (intersection of all sets)
    if all_layer_types:
        common_layers = set.intersection(*[layer_set for layer_set in all_layer_types.values()])
    else:
        common_layers = set()
    
    # Find unique layer types for each task
    unique_layers = {}
    for task, layer_set in all_layer_types.items():
        other_tasks = [t for t in all_layer_types if t != task]
        if other_tasks:
            other_layers = set.union(*[all_layer_types[t] for t in other_tasks])
            unique_to_task = layer_set - other_layers
        else:
            unique_to_task = layer_set
            
        if unique_to_task:
            unique_layers[task] = unique_to_task
    
    return common_layers, unique_layers

def compare_layer_distributions(task_dfs):
    """Compare layer type distributions between tasks"""
    layer_counts = {}
    
    for task, df in task_dfs.items():
        counts = df['Layer Type'].value_counts(normalize=True) * 100
        layer_counts[task] = counts
    
    # Convert to DataFrame for easier comparison
    comparison_df = pd.DataFrame(layer_counts)
    comparison_df = comparison_df.fillna(0)
    
    return comparison_df

def analyze_specialized_components(task_dfs):
    """Identify specialized components unique to specific tasks"""
    specialized = {}
    
    # Get all unique layer types across all tasks
    all_layers = set()
    task_layers = {}
    
    for task, df in task_dfs.items():
        layers = set(df['Layer Type'].unique())
        task_layers[task] = layers
        all_layers.update(layers)
    
    # Find layers unique to each task
    for task, layers in task_layers.items():
        other_tasks_layers = set()
        for t, l in task_layers.items():
            if t != task:
                other_tasks_layers.update(l)
                
        unique_layers = layers - other_tasks_layers
        if unique_layers:
            specialized[task] = unique_layers
    
    return specialized

def analyze_hierarchy(task_dfs):
    """Analyze hierarchical structures in model architectures"""
    hierarchy_patterns = {}
    
    for task, df in task_dfs.items():
        # Look for nested blocks and hierarchical patterns
        if 'Layer Name' in df.columns:
            # Extract hierarchical structure using layer names
            hierarchy = defaultdict(int)
            
            # Count components at different nesting levels
            for _, row in df.iterrows():
                layer_name = row['Layer Name']
                # Count the number of dots to determine nesting level
                nesting_level = layer_name.count('.')
                hierarchy[nesting_level] += 1
            
            hierarchy_patterns[task] = dict(hierarchy)
    
    return hierarchy_patterns 
"""
Neural Network Architecture Analyzer

This script analyzes the structure and patterns of deep neural network architectures
from processed model data, generating visualizations, statistics, and reports.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import glob
import shutil
import re
import argparse

# Define default input and output directories
DEFAULT_INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                      "model_architecture", "processed")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                       "analysis_result")

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural Network Architecture Analyzer')
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR,
                        help='Directory containing the processed model data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save analysis results. If not specified, a directory will be created based on the input directory name.')
    parser.add_argument('--append_timestamp', action='store_true',
                        help='Append a timestamp to the output directory name to avoid overwriting previous results')
    return parser.parse_args()

# Global variables for directories
INPUT_DIR = DEFAULT_INPUT_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

def create_output_dirs():
    """Create the output directory structure"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images", "boxplot"), exist_ok=True)  # Create dedicated folder for boxplots
    os.makedirs(os.path.join(OUTPUT_DIR, "csv_data"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "statistics_txt"), exist_ok=True)

def load_csv_files():
    """Load all CSV files from processed directory structure"""
    task_dfs = {}
    model_dfs = {}
    
    # Get all directory names (tasks)
    task_dirs = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    for task in task_dirs:
        task_path = os.path.join(INPUT_DIR, task)
        csv_files = glob.glob(os.path.join(task_path, "*.csv"))
        
        # Filter out deduplication_results.csv
        csv_files = [f for f in csv_files if os.path.basename(f) != "deduplication_results.csv"]
        
        if not csv_files:
            continue
            
        task_model_dfs = {}
        task_models = []
        
        # Load all CSV files for this task
        for file in csv_files:
            try:
                model_name = os.path.basename(file).replace('.csv', '')
                df = pd.read_csv(file)
                if 'Layer Type' in df.columns and 'Block' in df.columns:
                    df['Model'] = model_name
                    task_model_dfs[model_name] = df
                    task_models.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if task_models:
            # Combine all dataframes for this task
            task_dfs[task] = pd.concat(task_models, ignore_index=True)
            model_dfs[task] = task_model_dfs
    
    return task_dfs, model_dfs

# Basic Analysis Functions
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
    common_layers = set.intersection(*[layer_set for layer_set in all_layer_types.values()])
    
    # Find unique layer types for each task
    unique_layers = {}
    for task, layer_set in all_layer_types.items():
        unique_to_task = layer_set - set.union(*[s for t, s in all_layer_types.items() if t != task])
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
        other_tasks_layers = set().union(*[l for t, l in task_layers.items() if t != task])
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

# Advanced Analysis Functions
def analyze_layer_sequences(model_dfs):
    """Analyze typical layer sequences in models for each task"""
    task_sequences = {}
    
    for task, models in model_dfs.items():
        # For each model, extract the sequence of layer types
        model_sequences = {}
        
        for model_name, df in models.items():
            # Extract sequence of layer types
            layer_sequence = df['Layer Type'].tolist()
            # Store sequence
            model_sequences[model_name] = layer_sequence
        
        # Find common subsequences across models in this task
        task_sequences[task] = model_sequences
        
    return task_sequences

def extract_common_blocks(model_dfs):
    """Extract common functional blocks across models for each task"""
    task_blocks = {}
    
    for task, models in model_dfs.items():
        # Analyze block patterns for each model
        block_patterns = defaultdict(list)
        
        for model_name, df in models.items():
            # Group by Block column to find block structures
            if 'Block' in df.columns:
                blocks = df.groupby('Block')['Layer Type'].apply(list).to_dict()
                for block_name, layers in blocks.items():
                    if block_name != 'base' and not pd.isna(block_name):
                        block_patterns[block_name].append((model_name, layers))
        
        # Keep blocks that appear in multiple models
        common_blocks = {block: patterns for block, patterns in block_patterns.items() 
                         if len(set(model for model, _ in patterns)) > 1}
        
        task_blocks[task] = common_blocks
    
    return task_blocks

def analyze_model_complexity(task_dfs):
    """Analyze model complexity metrics by task"""
    complexity = {}
    
    for task, df in task_dfs.items():
        # Count unique models
        models_count = df['Model'].nunique()
        
        # Average number of layers per model
        avg_layers = df.groupby('Model').size().mean()
        
        # Layer type diversity
        layer_diversity = df['Layer Type'].nunique()
        
        # If parameters are available, analyze them
        params_info = None
        if 'Parameters' in df.columns:
            # Try to extract numeric parameter counts
            params = []
            pattern = r"'Input Features': (\d+), 'Output Features': (\d+)"
            
            for param_str in df['Parameters'].dropna():
                if isinstance(param_str, str):
                    matches = re.findall(pattern, param_str)
                    for match in matches:
                        if len(match) == 2:
                            in_features, out_features = int(match[0]), int(match[1])
                            params.append(in_features * out_features)
            
            if params:
                params_info = {
                    'total_params': sum(params),
                    'avg_params': np.mean(params),
                    'max_params': max(params)
                }
        
        complexity[task] = {
            'model_count': models_count,
            'avg_layers': avg_layers,
            'layer_diversity': layer_diversity,
            'params_info': params_info
        }
    
    return complexity

# Visualization Functions
def plot_layer_distributions(layer_type_counts, analysis_type=""):
    """Create visualizations for layer type distributions"""
    # Get top 20 most common layer types across all tasks
    all_counts = pd.concat([counts.rename(task) for task, counts in layer_type_counts.items()], axis=1)
    all_counts = all_counts.fillna(0)
    top_layers = all_counts.sum(axis=1).nlargest(20).index
    
    # Create a specialized plot for the top layer types with better proportions
    plt.figure(figsize=(16, 10))  # Better proportion for this specific chart
    plot_data = all_counts.loc[top_layers].T
    
    # Normalize by the total count per task
    normalized_data = plot_data.div(plot_data.sum(axis=1), axis=0) * 100
    
    # Create a stacked bar chart with custom styling
    ax = normalized_data.plot(kind='bar', stacked=True, width=0.7)  # Slightly thinner bars
    plt.title(f'Distribution of Top 20 Layer Types Across Tasks {analysis_type}', fontsize=16)
    plt.ylabel('Percentage', fontsize=14)
    plt.xlabel('Task', fontsize=14)
    plt.xticks(rotation=30, fontsize=12, ha='right')  # Less rotation for better readability
    plt.yticks(fontsize=12)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add a thin grid on the y-axis only
    plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    
    # Add percentage labels to show 0%, 20%, 40%, etc.
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x)}%"))
    
    # Better legend: place it outside the plot to the right with better styling
    plt.legend(
        bbox_to_anchor=(1.02, 0.35),
        loc='center left',
        fontsize=11,
        frameon=True,
        framealpha=0.95,
        edgecolor='lightgray',
        title="Layer Types",
        title_fontsize=12
    )
    
    # Create more space for the legend
    plt.subplots_adjust(right=0.7)
    
    # Ensure y axis goes from 0 to 100
    plt.ylim(0, 100)
    
    # Add thin horizontal lines at 20% intervals
    for y in range(20, 101, 20):
        plt.axhline(y=y, color='gray', linestyle='--', alpha=0.3, zorder=0)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'images', 'layer_type_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a heatmap for the top layer types with log scale
    plt.figure(figsize=(20, 16))
    
    # Apply natural log transform to data, but need to handle 0 values
    # First replace all 0 values with a small value (e.g., 0.1) to avoid log(0) issues
    plot_data_for_heatmap = plot_data.copy()
    plot_data_for_heatmap = plot_data_for_heatmap.replace(0, 0.1)
    # Then apply natural log transformation
    log_data = np.log(plot_data_for_heatmap)
    
    # Use log-transformed data to draw heatmap
    sns.heatmap(log_data.T, annot=False, cmap='YlGnBu', fmt='.2f')
    plt.title(f'Heatmap of Top 20 Layer Types Across Tasks (Log Scale) {analysis_type}', fontsize=18)
    plt.xlabel('Task', fontsize=14)
    plt.ylabel('Layer Type', fontsize=14)
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'images', 'layer_type_heatmap_log.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Keep the original heatmap for comparison
    plt.figure(figsize=(20, 16))
    sns.heatmap(plot_data.T, annot=False, cmap='YlGnBu', fmt='.0f')
    plt.title(f'Heatmap of Top 20 Layer Types Across Tasks (Original Scale) {analysis_type}', fontsize=18)
    plt.xlabel('Task', fontsize=14)
    plt.ylabel('Layer Type', fontsize=14)
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'images', 'layer_type_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Export raw data to a single CSV file instead of multiple
    all_counts.to_csv(os.path.join(OUTPUT_DIR, 'csv_data', 'layer_distribution_data.csv'))

def plot_task_comparison(layer_comparison, analysis_type=""):
    """Visualize layer type distribution comparison between tasks"""
    # Get top 20 layers across all tasks
    top_layers = layer_comparison.sum(axis=1).nlargest(20).index
    comparison_data = layer_comparison.loc[top_layers]
    
    # Create heatmap with increased size
    plt.figure(figsize=(18, 14))  # Increased figure size
    sns.heatmap(comparison_data, annot=True, cmap='viridis', fmt='.1f', annot_kws={"size": 10})
    plt.title(f'Layer Type Distribution Comparison Between Tasks (%) {analysis_type}', fontsize=18)
    plt.xlabel('Task', fontsize=14)
    plt.ylabel('Layer Type', fontsize=14)
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'images', 'task_comparison_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Export data
    comparison_data.to_csv(os.path.join(OUTPUT_DIR, 'csv_data', 'task_comparison.csv'))

def visualize_hierarchy(hierarchy_patterns, analysis_type=""):
    """Visualize the hierarchical patterns across tasks"""
    if not hierarchy_patterns:
        print("No hierarchy data available")
        return
        
    # Create DataFrame for visualization
    data = []
    for task, levels in hierarchy_patterns.items():
        for level, count in levels.items():
            data.append({'Task': task, 'Nesting Level': level, 'Count': count})
    
    df = pd.DataFrame(data)
    
    # Create visualization with increased size
    plt.figure(figsize=(16, 10))  # Increased figure size
    sns.barplot(x='Task', y='Count', hue='Nesting Level', data=df)
    plt.title(f'Model Hierarchy by Task {analysis_type}', fontsize=18)
    plt.xlabel('Task', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, fontsize=12, ha='right')
    plt.yticks(fontsize=12)
    plt.legend(title='Nesting Level', title_fontsize=12, fontsize=11, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'images', "hierarchy_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Export raw data - directly to CSV without pivot
    df.to_csv(os.path.join(OUTPUT_DIR, 'csv_data', 'hierarchy_data.csv'), index=False)

def visualize_complexity(complexity, analysis_type=""):
    """Visualize model complexity comparison"""
    # Extract data for visualization
    tasks = list(complexity.keys())
    avg_layers = [data['avg_layers'] for data in complexity.values()]
    diversity = [data['layer_diversity'] for data in complexity.values()]
    
    # Create a DataFrame for all complexity metrics
    complexity_df = pd.DataFrame({
        'Task': tasks,
        'Average_Layers': avg_layers,
        'Layer_Diversity': diversity,
        'Model_Count': [data['model_count'] for data in complexity.values()],
    })
    
    # Add parameter info where available
    for i, task in enumerate(tasks):
        params_info = complexity[task]['params_info']
        if params_info:
            complexity_df.loc[i, 'Total_Parameters'] = params_info['total_params']
            complexity_df.loc[i, 'Avg_Parameters'] = params_info['avg_params']
    
    # Create a simple bar chart for average layers per model
    plt.figure(figsize=(16, 9))  # Increased figure size
    plt.bar(tasks, avg_layers, color='skyblue')
    plt.title(f'Average Layers per Model by Task {analysis_type}', fontsize=18)
    plt.ylabel('Average Number of Layers', fontsize=14)
    plt.xlabel('Task', fontsize=14)
    plt.xticks(rotation=45, fontsize=12, ha='right')
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'images', "avg_layers_by_task.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create bar chart for layer type diversity
    plt.figure(figsize=(16, 9))  # Increased figure size
    plt.bar(tasks, diversity, color='lightgreen')
    plt.title(f'Layer Type Diversity by Task {analysis_type}', fontsize=18)
    plt.ylabel('Number of Unique Layer Types', fontsize=14)
    plt.xlabel('Task', fontsize=14)
    plt.xticks(rotation=45, fontsize=12, ha='right')
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'images', "layer_diversity_by_task.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Export single consolidated complexity data file
    complexity_df.to_csv(os.path.join(OUTPUT_DIR, 'csv_data', 'model_complexity.csv'), index=False)

def visualize_nesting_levels(hierarchy_patterns, analysis_type=""):
    """Visualize the median and mean nesting levels across tasks as line charts"""
    if not hierarchy_patterns:
        print("No hierarchy data available for nesting level visualization")
        return
        
    # Extract nesting level data and calculate statistics
    tasks = []
    medians = []
    means = []
    
    for task, levels in hierarchy_patterns.items():
        if levels:  # Check if the dictionary is not empty
            # Convert the level data into a list of nesting levels
            nesting_data = []
            for level, count in levels.items():
                # Repeat each nesting level 'count' times
                nesting_data.extend([level] * count)
            
            if nesting_data:  # Check if we have data to calculate statistics
                tasks.append(task)
                medians.append(np.median(nesting_data))
                means.append(np.mean(nesting_data))
    
    # Sort the data by task name for consistency
    task_indices = np.argsort(tasks)
    sorted_tasks = [tasks[i] for i in task_indices]
    sorted_medians = [medians[i] for i in task_indices]
    sorted_means = [means[i] for i in task_indices]
    
    # Create median nesting level line chart with increased size
    plt.figure(figsize=(16, 9))  # Increased figure size
    plt.plot(sorted_tasks, sorted_medians, marker='o', linestyle='-', linewidth=2, markersize=10)
    plt.title(f'Median Nesting Level by Task {analysis_type}', fontsize=18)
    plt.xlabel('Task', fontsize=14)
    plt.ylabel('Median Nesting Level', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, fontsize=12, ha='right')
    plt.yticks(fontsize=12)
    
    # Add data labels on points
    for i, (x, y) in enumerate(zip(sorted_tasks, sorted_medians)):
        plt.text(i, y + 0.05, f'{y:.2f}', ha='center', va='bottom', fontsize=11)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'images', "median_nesting_level.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create mean nesting level line chart with increased size
    plt.figure(figsize=(16, 9))  # Increased figure size
    plt.plot(sorted_tasks, sorted_means, marker='o', linestyle='-', linewidth=2, color='green', markersize=10)
    plt.title(f'Mean Nesting Level by Task {analysis_type}', fontsize=18)
    plt.xlabel('Task', fontsize=14)
    plt.ylabel('Mean Nesting Level', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, fontsize=12, ha='right')
    plt.yticks(fontsize=12)
    
    # Add data labels on points
    for i, (x, y) in enumerate(zip(sorted_tasks, sorted_means)):
        plt.text(i, y + 0.05, f'{y:.2f}', ha='center', va='bottom', fontsize=11)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'images', "mean_nesting_level.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Export the nesting level statistics to CSV
    nesting_stats = pd.DataFrame({
        'Task': sorted_tasks,
        'Median_Nesting_Level': sorted_medians,
        'Mean_Nesting_Level': sorted_means
    })
    nesting_stats.to_csv(os.path.join(OUTPUT_DIR, 'csv_data', 'nesting_level_statistics.csv'), index=False)

def plot_layer_boxplots_per_task(model_dfs, analysis_type=""):
    """
    Create layer type boxplots for each task
    One figure per task, showing the distribution of top 10 layer types across different models
    """
    print("Generating layer distribution boxplots for each task...")
    
    for task, models in model_dfs.items():
        # Skip tasks with insufficient data
        if len(models) < 3:
            print(f"Skipping boxplot for task {task}: not enough models")
            continue
            
        # Calculate layer type counts for all models in this task
        all_layer_counts = {}
        for model_name, df in models.items():
            layer_counts = df['Layer Type'].value_counts()
            for layer_type, count in layer_counts.items():
                if layer_type not in all_layer_counts:
                    all_layer_counts[layer_type] = []
                all_layer_counts[layer_type].append(count)
        
        # Find the top 10 layer types in this task
        layer_type_totals = {lt: sum(counts) for lt, counts in all_layer_counts.items()}
        top_layer_types = sorted(layer_type_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        top_layer_names = [lt for lt, _ in top_layer_types]
        
        # Prepare data for each top layer type across all models
        # If a model doesn't have this layer type, count it as 0
        boxplot_data = []
        for layer_type in top_layer_names:
            counts = all_layer_counts.get(layer_type, [])
            # Ensure all models have data points
            if len(counts) < len(models):
                # Add missing models (with value 0)
                missing_count = len(models) - len(counts)
                counts.extend([0] * missing_count)
            boxplot_data.append(counts)
        
        # Create boxplot with increased width to avoid overlap
        plt.figure(figsize=(16, 10))  # Increase figure size
        
        # Use pandas DataFrame to organize data for easier boxplot creation
        boxplot_df = pd.DataFrame(data={layer: counts for layer, counts in zip(top_layer_names, boxplot_data)})
        
        # Determine Y-axis upper limit, intelligently adjusted based on data distribution
        max_value = boxplot_df.max().max()
        # Calculate 95th percentile to avoid extreme values dominating the chart
        percentile_95 = np.percentile(boxplot_df.values.flatten(), 95)
        # Use 1.5x of 95th percentile or 1.2x of max value (whichever is smaller) as Y-axis limit
        y_limit = min(percentile_95 * 1.5, max_value * 1.2)
        
        # Create boxplot with more appealing style
        # Reduce box width to increase spacing
        ax = sns.boxplot(
            data=boxplot_df, 
            palette="pastel",  # More subtle color palette
            linewidth=1.5, 
            fliersize=4,       # Increase outlier point size
            width=0.6,         # Reduce box width
            showfliers=True,   # Show outliers
            showmeans=True,    # Show mean values
            meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"7"}  # Mean point style
        )
        
        # Add scatter plot to show actual data points, with reduced opacity and size to minimize visual clutter
        for i, layer in enumerate(boxplot_df.columns):
            # Greater jitter to reduce overlap
            x = np.random.normal(i, 0.15, size=len(boxplot_df[layer]))
            # Only show non-zero values to reduce visual clutter
            non_zero_mask = boxplot_df[layer] > 0
            if non_zero_mask.any():  # Only plot if there are non-zero values
                plt.scatter(
                    x[non_zero_mask], 
                    boxplot_df[layer][non_zero_mask], 
                    color='navy', 
                    alpha=0.3,  # Lower opacity
                    s=15,       # Increase point size
                    edgecolor='none'  # Remove point edge lines
                )
            
        # Use better chart styling
        plt.title(f'Distribution of Top 10 Layer Types Across Models in "{task}" {analysis_type}', fontsize=16)
        plt.ylabel('Count per Model', fontsize=14)
        plt.xlabel('Layer Type', fontsize=14)
        plt.xticks(rotation=30, fontsize=12, ha='right')  # Reduce rotation angle
        plt.yticks(fontsize=12)
        
        # Set Y-axis upper limit
        if y_limit > 0:  # Ensure there's a reasonable upper limit
            plt.ylim(0, y_limit)
            
        # Add clearer grid lines
        plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
        
        # Remove top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add slight shadow under boxplots to improve readability
        for i, patch in enumerate(ax.patches):
            # Only add shadow effect to box elements
            if i % 6 < 5:  # Filter out non-box elements
                patch.set_zorder(2)  # Ensure in front of grid
        
        # Save chart
        plt.tight_layout()
        safe_task_name = task.replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(OUTPUT_DIR, 'images', 'boxplot', f"{safe_task_name}_layer_boxplot.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save raw data
        boxplot_df.to_csv(os.path.join(OUTPUT_DIR, 'csv_data', f"{safe_task_name}_layer_boxplot_data.csv"))
        
    print(f"Layer boxplots saved to {os.path.join(OUTPUT_DIR, 'images', 'boxplot')}")

# Report Generation Functions
def create_architecture_patterns_report(task_dfs, hierarchy_patterns, specialized_components, analysis_type=""):
    """Generate a report on architecture patterns"""
    report_lines = []
    report_lines.append(f"# Neural Network Architecture Patterns {analysis_type}")
    report_lines.append("\n## Task Overview")
    report_lines.append(f"Total tasks analyzed: {len(task_dfs)}")
    report_lines.append(f"Tasks: {', '.join(sorted(task_dfs.keys()))}")
    
    # Architectural patterns
    report_lines.append("\n## Hierarchical Structures")
    for task, levels in hierarchy_patterns.items():
        report_lines.append(f"\n### {task}")
        report_lines.append("| Nesting Level | Component Count | Percentage |")
        report_lines.append("|--------------|----------------|------------|")
        total = sum(levels.values())
        for level, count in sorted(levels.items()):
            percentage = count / total * 100
            report_lines.append(f"| {level} | {count} | {percentage:.1f}% |")
    
    # Specialized components
    report_lines.append("\n## Task-Specific Components")
    for task, components in specialized_components.items():
        report_lines.append(f"\n### {task}")
        report_lines.append("| Layer Type |")
        report_lines.append("|-----------|")
        for component in sorted(components):
            report_lines.append(f"| {component} |")
    
    # Write to file
    with open(os.path.join(OUTPUT_DIR, 'reports', 'architecture_patterns.md'), 'w') as f:
        f.write('\n'.join(report_lines))

def generate_task_summaries(task_dfs, model_dfs, analysis_type=""):
    """Generate summary reports for each task"""
    for task, df in task_dfs.items():
        summary_lines = []
        summary_lines.append(f"# {task} Architecture Summary {analysis_type}")
        
        # Basic metrics
        model_count = df['Model'].nunique()
        layer_count = len(df)
        layer_types = df['Layer Type'].nunique()
        
        summary_lines.append(f"\n## Overview")
        summary_lines.append(f"- Models analyzed: {model_count}")
        summary_lines.append(f"- Total layers: {layer_count}")
        summary_lines.append(f"- Unique layer types: {layer_types}")
        
        # Top layer types
        summary_lines.append(f"\n## Layer Type Distribution")
        summary_lines.append("| Layer Type | Count | Percentage |")
        summary_lines.append("|-----------|-------|------------|")
        
        layer_counts = df['Layer Type'].value_counts()
        top20 = layer_counts.nlargest(20)
        for layer, count in top20.items():
            percentage = count / len(df) * 100
            summary_lines.append(f"| {layer} | {count} | {percentage:.1f}% |")
        
        # Sample models
        summary_lines.append(f"\n## Sample Models")
        for model_name, model_df in list(model_dfs[task].items())[:3]:  # Take up to 3 models
            summary_lines.append(f"\n### {model_name}")
            summary_lines.append(f"- Layers: {len(model_df)}")
            
            # Show top 5 layers
            summary_lines.append("\nLayer sequence (first 5):")
            summary_lines.append("```")
            for i, (_, layer) in enumerate(model_df.head(5).iterrows()):
                summary_lines.append(f"{i+1}. {layer['Layer Type']}")
            summary_lines.append("```")
        
        # Write to file
        with open(os.path.join(OUTPUT_DIR, 'reports', f'{task}_summary.md'), 'w') as f:
            f.write('\n'.join(summary_lines))

def export_comparative_data(task_dfs):
    """Export comparative data for further analysis - consolidated into fewer files"""
    # Create a consolidated summary rather than per-task files
    
    # 1. Single consolidated layer types file instead of per-task files
    all_layer_types = {}
    for task, df in task_dfs.items():
        all_layer_types[task] = df['Layer Type'].value_counts()
    
    # Combine into a single DataFrame
    layer_types_df = pd.DataFrame(all_layer_types)
    layer_types_df = layer_types_df.fillna(0)
    # Export as a single file
    layer_types_df.to_csv(os.path.join(OUTPUT_DIR, 'csv_data', 'all_tasks_layer_types.csv'))
    
    # 2. Consolidated model metrics instead of individual numeric summaries
    model_metrics = []
    for task, df in task_dfs.items():
        # Get key metrics per model
        for model, model_df in df.groupby('Model'):
            metrics = {
                'Task': task,
                'Model': model,
                'Layer_Count': len(model_df),
                'Unique_Layer_Types': model_df['Layer Type'].nunique(),
                'Block_Count': model_df['Block'].nunique() if 'Block' in model_df.columns else 0,
            }
            model_metrics.append(metrics)
    
    # Create and export a single consolidated metrics file
    if model_metrics:
        metrics_df = pd.DataFrame(model_metrics)
        metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'csv_data', 'model_metrics_summary.csv'), index=False)

def export_block_statistics(common_blocks):
    """Export statistics about common blocks to CSV and TXT files"""
    # Create a summary DataFrame
    data = []
    for task, blocks in common_blocks.items():
        for block_name, instances in blocks.items():
            data.append({
                'Task': task,
                'Block Name': block_name,
                'Model Count': len(set(model for model, _ in instances)),
                'Average Length': np.mean([len(layers) for _, layers in instances])
            })
    
    if data:
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(OUTPUT_DIR, 'csv_data', 'common_blocks.csv'), index=False)
        
        # Also create a text summary
        with open(os.path.join(OUTPUT_DIR, 'reports', 'common_blocks.txt'), 'w') as f:
            f.write("Common Functional Blocks by Task\n\n")
            for task, blocks in common_blocks.items():
                if blocks:
                    f.write(f"{task}:\n")
                    for block_name, instances in blocks.items():
                        model_count = len(set(model for model, _ in instances))
                        f.write(f"  {block_name}: found in {model_count} models\n")
                    f.write("\n")

def export_statistics_as_txt(layer_type_counts, block_counts, hierarchy_patterns, complexity, common_layers, unique_layers):
    """Export various statistics to txt files for easy reference"""
    txt_dir = os.path.join(OUTPUT_DIR, "statistics_txt")
    os.makedirs(txt_dir, exist_ok=True)
    
    # Export common layers
    with open(os.path.join(txt_dir, "common_layers.txt"), "w") as f:
        f.write("Common layer types across all tasks:\n")
        for layer in sorted(common_layers):
            f.write(f"{layer}\n")
    
    # Export unique layers by task
    with open(os.path.join(txt_dir, "unique_layers_by_task.txt"), "w") as f:
        f.write("Unique layer types by task:\n\n")
        for task, layers in unique_layers.items():
            f.write(f"{task}:\n")
            for layer in sorted(layers):
                f.write(f"  {layer}\n")
            f.write("\n")
    
    # Export top layer types by task
    with open(os.path.join(txt_dir, "top_layer_types.txt"), "w") as f:
        f.write("Top 20 layer types by task:\n\n")
        for task, counts in layer_type_counts.items():
            f.write(f"{task}:\n")
            top20 = counts.nlargest(20)
            for layer, count in top20.items():
                f.write(f"  {layer}: {count} ({count/sum(counts)*100:.1f}%)\n")
            f.write("\n")
    
    # Export top blocks by task
    with open(os.path.join(txt_dir, "top_blocks.txt"), "w") as f:
        f.write("Top 20 blocks by task:\n\n")
        for task, counts in block_counts.items():
            if not counts.empty:
                f.write(f"{task}:\n")
                top20 = counts.nlargest(20)
                for block, count in top20.items():
                    f.write(f"  {block}: {count} ({count/sum(counts)*100:.1f}%)\n")
                f.write("\n")

def create_index_file():
    """Create an index file listing all generated files with brief descriptions"""
    # Organize files by moving them to subdirectories
    images_dir = os.path.join(OUTPUT_DIR, "images")
    csv_dir = os.path.join(OUTPUT_DIR, "csv_data")
    report_dir = os.path.join(OUTPUT_DIR, "reports")
    
    # Create the index file
    index_file = os.path.join(OUTPUT_DIR, "analysis_index.md")
    
    with open(index_file, "w") as f:
        f.write("# Model Architecture Analysis Results Index\n\n")
        
        # Visualizations section
        f.write("## Visualizations\n\n")
        if os.path.exists(images_dir) and os.listdir(images_dir):
            for img in sorted(os.listdir(images_dir)):
                img_path = os.path.join("images", img)
                name = os.path.splitext(img)[0].replace("_", " ").title()
                f.write(f"- [{name}]({img_path})\n")
        
        # Data files section
        f.write("\n## Data Files (CSV)\n\n")
        if os.path.exists(csv_dir) and os.listdir(csv_dir):
            for csv in sorted(os.listdir(csv_dir)):
                csv_path = os.path.join("csv_data", csv)
                name = os.path.splitext(csv)[0].replace("_", " ").title()
                f.write(f"- [{name}]({csv_path})\n")
        
        # Reports section
        f.write("\n## Analysis Reports\n\n")
        if os.path.exists(report_dir) and os.listdir(report_dir):
            md_files = [f for f in os.listdir(report_dir) if f.endswith('.md')]
            txt_files = [f for f in os.listdir(report_dir) if f.endswith('.txt')]
            
            for md in sorted(md_files):
                md_path = os.path.join("reports", md)
                name = os.path.splitext(md)[0].replace("_", " ").title()
                f.write(f"- [{name}]({md_path})\n")
            
            if txt_files:
                f.write("\n### Detailed Statistics\n\n")
                for txt in sorted(txt_files):
                    txt_path = os.path.join("reports", txt)
                    name = os.path.splitext(txt)[0].replace("_", " ").title()
                    f.write(f"- [{name}]({txt_path})\n")
    
    print(f"Index file created: {index_file}")

def main():
    """Main function to run the neural network architecture analyzer"""
    global INPUT_DIR, OUTPUT_DIR
    
    # Parse command line arguments
    args = parse_arguments()
    INPUT_DIR = args.input_dir
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} does not exist.")
        return
    
    # Set up output directory based on input directory if not specified
    if args.output_dir is None:
        # Get the base name of the input directory
        input_dirname = os.path.basename(os.path.normpath(INPUT_DIR))
        OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            f"analysis_result_{input_dirname}")
    else:
        OUTPUT_DIR = args.output_dir
    
    # Add timestamp if requested
    if args.append_timestamp:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_DIR = f"{OUTPUT_DIR}_{timestamp}"
    
    # Determine analysis type based on input directory name
    analysis_type = ""
    if "process_with_name" in INPUT_DIR.lower():
        analysis_type = "(name-based)"
    elif "process_with_architecture" in INPUT_DIR.lower() or "process_with_arc" in INPUT_DIR.lower():
        analysis_type = "(arc-based)"
    
    print("Neural Network Architecture Analyzer")
    print("===================================")
    
    # Create output directories
    create_output_dirs()
    print(f"Input directory: {INPUT_DIR}")
    print(f"Analysis type: {analysis_type if analysis_type else 'standard'}")
    print(f"Output will be saved to: {OUTPUT_DIR}")
    
    # Load data
    print("\nLoading model architecture data...")
    task_dfs, model_dfs = load_csv_files()
    
    if not task_dfs:
        print("No data found to analyze. Exiting.")
        return
    
    task_count = len(task_dfs)
    model_count = sum(len(models) for models in model_dfs.values())
    print(f"Loaded data for {task_count} tasks and {model_count} models.")
    
    # Pass analysis type to the visualization functions
    visualize_with_type(task_dfs, model_dfs, analysis_type)

# New function to handle all visualizations with the analysis type
def visualize_with_type(task_dfs, model_dfs, analysis_type):
    """Run all visualizations with the specified analysis type in titles"""
    print("\nPerforming analysis and generating visualizations...")
    
    # Basic analysis
    layer_type_counts = analyze_layer_types(task_dfs)
    block_counts = analyze_blocks(task_dfs)
    common_layers, unique_layers = common_unique_layers(task_dfs)
    layer_distribution_comparison = compare_layer_distributions(task_dfs)
    specialized_components = analyze_specialized_components(task_dfs)
    hierarchy_patterns = analyze_hierarchy(task_dfs)
    
    # Print some basic results
    print(f"Found {len(common_layers)} common layer types across all tasks:")
    if common_layers:
        print(", ".join(sorted(common_layers)))
    
    # Advanced analysis
    task_sequences = analyze_layer_sequences(model_dfs)
    common_blocks = extract_common_blocks(model_dfs)
    complexity = analyze_model_complexity(task_dfs)
    
    # Modified visualization calls with analysis type
    plot_layer_distributions(layer_type_counts, analysis_type)
    plot_task_comparison(layer_distribution_comparison, analysis_type)
    visualize_hierarchy(hierarchy_patterns, analysis_type)
    visualize_complexity(complexity, analysis_type)
    visualize_nesting_levels(hierarchy_patterns, analysis_type)
    plot_layer_boxplots_per_task(model_dfs, analysis_type)
    
    # Export consolidated statistics
    print("\nExporting consolidated statistics...")
    export_block_statistics(common_blocks)
    export_statistics_as_txt(layer_type_counts, block_counts, hierarchy_patterns, 
                           complexity, common_layers, unique_layers)
    export_comparative_data(task_dfs)
    
    # Generate reports
    print("\nGenerating reports...")
    create_architecture_patterns_report(task_dfs, hierarchy_patterns, specialized_components, analysis_type)
    generate_task_summaries(task_dfs, model_dfs, analysis_type)
    
    # Create index file
    create_index_file()

if __name__ == "__main__":
    main() 
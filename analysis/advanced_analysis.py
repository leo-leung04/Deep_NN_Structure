"""
Advanced analysis and visualization functions for model architecture analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from .config import OUTPUT_DIR

def plot_layer_distributions(layer_type_counts):
    """
    Plot layer type distributions for each task
    
    Args:
        layer_type_counts: Dictionary of layer type counts by task
    """
    if not layer_type_counts:
        print("No layer type counts provided for plotting")
        return
    
    # Create the images directory if it doesn't exist
    img_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    # Plot for each task
    for task, counts in layer_type_counts.items():
        if len(counts) == 0:
            continue
            
        plt.figure(figsize=(12, 8))
        
        # Sort by count for better visualization
        top_n = 15
        top_counts = counts.nlargest(top_n)
        
        ax = sns.barplot(x=top_counts.index, y=top_counts.values, palette='viridis')
        
        plt.title(f"Top {top_n} Layer Types for {task}")
        plt.xlabel("Layer Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(img_dir, f"{task}_layer_types.png"))
        plt.close()

def plot_task_comparison(comparison_df):
    """
    Plot comparison of layer type distributions across tasks
    
    Args:
        comparison_df: DataFrame with tasks as columns and layer types as index
    """
    if comparison_df.empty:
        print("No comparison data provided for plotting")
        return
        
    img_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    # Get common layers across all tasks
    all_tasks = comparison_df.columns
    common_layers = []
    
    for layer in comparison_df.index:
        # Check if layer exists in all tasks
        if not comparison_df.loc[layer].isna().any():
            common_layers.append(layer)
    
    # Filter to common layers with significant presence
    if common_layers:
        common_df = comparison_df.loc[common_layers]
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(common_df, annot=True, cmap="YlGnBu", fmt=".1f")
        plt.title("Layer Type Distribution Comparison Across Tasks (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, "task_layer_comparison.png"))
        plt.close()
    
    # Create a bar chart for comparison
    plt.figure(figsize=(15, 10))
    
    # Select top 10 layers by mean presence
    top_layers = comparison_df.fillna(0).mean(axis=1).nlargest(10).index
    top_df = comparison_df.loc[top_layers]
    
    # Create a grouped bar chart
    x = np.arange(len(top_layers))
    width = 0.8 / len(all_tasks)
    
    for i, task in enumerate(all_tasks):
        offset = i * width - (len(all_tasks) * width / 2) + width/2
        plt.bar(x + offset, top_df[task], width, label=task)
    
    plt.xlabel('Layer Type')
    plt.ylabel('Percentage (%)')
    plt.title('Top Layer Types Comparison Across Tasks')
    plt.xticks(x, top_layers, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(img_dir, "task_top_layers_comparison.png"))
    plt.close()

def create_architecture_patterns_report(task_dfs, hierarchy_patterns, specialized_components):
    """
    Create a report on architectural patterns found in each task
    
    Args:
        task_dfs: Dictionary of DataFrames by task
        hierarchy_patterns: Dictionary of hierarchy patterns by task
        specialized_components: Dictionary of specialized components by task
    """
    reports_dir = os.path.join(OUTPUT_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    report_path = os.path.join(reports_dir, "architecture_patterns.md")
    
    with open(report_path, 'w') as f:
        f.write("# Neural Network Architecture Patterns Analysis\n\n")
        
        # Overall statistics
        f.write("## Overall Statistics\n\n")
        f.write(f"- Total tasks analyzed: {len(task_dfs)}\n")
        f.write(f"- Tasks: {', '.join(task_dfs.keys())}\n\n")
        
        # Hierarchy patterns
        f.write("## Hierarchy Patterns\n\n")
        f.write("This section shows the distribution of components at different nesting levels in the models.\n\n")
        
        for task, hierarchy in hierarchy_patterns.items():
            f.write(f"### {task}\n\n")
            f.write("| Nesting Level | Component Count |\n")
            f.write("|---------------|----------------|\n")
            
            for level, count in sorted(hierarchy.items()):
                f.write(f"| {level} | {count} |\n")
            
            f.write("\n")
        
        # Specialized components
        f.write("## Task-Specific Components\n\n")
        f.write("This section lists components that are unique to specific tasks.\n\n")
        
        for task, components in specialized_components.items():
            f.write(f"### {task}\n\n")
            f.write("Unique components:\n\n")
            for component in sorted(components):
                f.write(f"- {component}\n")
            f.write("\n")
    
    print(f"Architecture patterns report saved to {report_path}")

def generate_task_summaries(task_dfs, model_dfs):
    """
    Generate summaries for each task and save to files
    
    Args:
        task_dfs: Dictionary of DataFrames by task
        model_dfs: Dictionary of model DataFrames by task
    """
    stats_dir = os.path.join(OUTPUT_DIR, "statistics_txt")
    os.makedirs(stats_dir, exist_ok=True)
    
    for task, df in task_dfs.items():
        summary_path = os.path.join(stats_dir, f"{task}_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write(f"Summary for Task: {task}\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            models_count = len(model_dfs.get(task, {}))
            f.write(f"Number of Models: {models_count}\n")
            
            layer_types = df['Layer Type'].nunique()
            f.write(f"Number of Unique Layer Types: {layer_types}\n")
            
            block_types = df['Block'].nunique()
            f.write(f"Number of Unique Block Types: {block_types}\n\n")
            
            # Most common layer types
            f.write("Most Common Layer Types:\n")
            top_layers = df['Layer Type'].value_counts().nlargest(10)
            for layer, count in top_layers.items():
                f.write(f"  - {layer}: {count}\n")
            f.write("\n")
            
            # Most common blocks
            f.write("Most Common Blocks:\n")
            top_blocks = df['Block'].value_counts().nlargest(10)
            for block, count in top_blocks.items():
                f.write(f"  - {block}: {count}\n")
            f.write("\n")
            
            # Average model complexity
            avg_layers = df.groupby('Model').size().mean()
            f.write(f"Average Layers per Model: {avg_layers:.2f}\n")
            
            # Save architectural trends
            f.write("\nArchitectural Trends:\n")
            
            # Calculate layer type percentages
            layer_percentages = df['Layer Type'].value_counts(normalize=True) * 100
            f.write("Layer Type Distribution (%):\n")
            for layer, pct in layer_percentages.nlargest(15).items():
                f.write(f"  - {layer}: {pct:.2f}%\n")
    
    print(f"Task summaries generated in {stats_dir}")

def export_comparative_data(task_dfs):
    """
    Export comparative data between tasks for further analysis
    
    Args:
        task_dfs: Dictionary of DataFrames by task
    """
    csv_dir = os.path.join(OUTPUT_DIR, "csv_data")
    os.makedirs(csv_dir, exist_ok=True)
    
    # Layer type distribution comparison
    layer_distribution = {}
    for task, df in task_dfs.items():
        layer_distribution[task] = df['Layer Type'].value_counts(normalize=True) * 100
    
    comparison_df = pd.DataFrame(layer_distribution)
    comparison_df = comparison_df.fillna(0)
    comparison_df.to_csv(os.path.join(csv_dir, "layer_distribution_comparison.csv"))
    
    # Block distribution comparison
    block_distribution = {}
    for task, df in task_dfs.items():
        block_distribution[task] = df['Block'].value_counts(normalize=True) * 100
    
    block_comparison_df = pd.DataFrame(block_distribution)
    block_comparison_df = block_comparison_df.fillna(0)
    block_comparison_df.to_csv(os.path.join(csv_dir, "block_distribution_comparison.csv"))
    
    print(f"Comparative data exported to {csv_dir}") 
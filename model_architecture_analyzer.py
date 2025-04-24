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

# Define input and output directories
INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                      "model_architecture", "processed")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                       "analysis_result")

def create_output_dirs():
    """Create the output directory structure"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
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
def plot_layer_distributions(layer_type_counts):
    """Create visualizations for layer type distributions"""
    # Get top 15 most common layer types across all tasks
    all_counts = pd.concat([counts.rename(task) for task, counts in layer_type_counts.items()], axis=1)
    all_counts = all_counts.fillna(0)
    top_layers = all_counts.sum(axis=1).nlargest(15).index
    
    # Create a plot for the top layer types
    plt.figure(figsize=(14, 8))
    plot_data = all_counts.loc[top_layers].T
    
    # Normalize by the total count per task
    normalized_data = plot_data.div(plot_data.sum(axis=1), axis=0) * 100
    
    # Create a stacked bar chart
    ax = normalized_data.plot(kind='bar', stacked=True)
    plt.title('Distribution of Top Layer Types Across Tasks (Percentage)')
    plt.ylabel('Percentage')
    plt.xlabel('Task')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'images', 'layer_type_distribution.png'))
    plt.close()
    
    # Create a heatmap for the top layer types
    plt.figure(figsize=(15, 10))
    sns.heatmap(plot_data.T, annot=False, cmap='YlGnBu', fmt='.0f')
    plt.title('Heatmap of Top Layer Types Across Tasks')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'images', 'layer_type_heatmap.png'))
    plt.close()
    
    # Export raw data to CSV
    normalized_data.to_csv(os.path.join(OUTPUT_DIR, 'csv_data', 'layer_type_distribution.csv'))
    plot_data.to_csv(os.path.join(OUTPUT_DIR, 'csv_data', 'layer_type_counts.csv'))

def plot_task_comparison(layer_comparison):
    """Visualize layer type distribution comparison between tasks"""
    # Get top 20 layers across all tasks
    top_layers = layer_comparison.sum(axis=1).nlargest(20).index
    comparison_data = layer_comparison.loc[top_layers]
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(comparison_data, annot=True, cmap='viridis', fmt='.1f')
    plt.title('Layer Type Distribution Comparison Between Tasks (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'images', 'task_comparison_heatmap.png'))
    plt.close()
    
    # Export data
    comparison_data.to_csv(os.path.join(OUTPUT_DIR, 'csv_data', 'task_comparison.csv'))

def visualize_hierarchy(hierarchy_patterns):
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
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Task', y='Count', hue='Nesting Level', data=df)
    plt.title('Model Hierarchy by Task')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'images', "hierarchy_comparison.png"))
    plt.close()
    
    # Export raw data to CSV
    pivot_data = df.pivot(index='Task', columns='Nesting Level', values='Count').fillna(0)
    pivot_data.to_csv(os.path.join(OUTPUT_DIR, 'csv_data', 'hierarchy_levels.csv'))

def visualize_complexity(complexity):
    """Visualize model complexity comparison"""
    # Extract data for visualization
    tasks = list(complexity.keys())
    avg_layers = [data['avg_layers'] for data in complexity.values()]
    diversity = [data['layer_diversity'] for data in complexity.values()]
    
    # Create a simple bar chart for average layers per model
    plt.figure(figsize=(10, 6))
    plt.bar(tasks, avg_layers, color='skyblue')
    plt.title('Average Layers per Model by Task')
    plt.ylabel('Average Number of Layers')
    plt.xlabel('Task')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'images', "avg_layers_by_task.png"))
    plt.close()
    
    # Create bar chart for layer type diversity
    plt.figure(figsize=(10, 6))
    plt.bar(tasks, diversity, color='lightgreen')
    plt.title('Layer Type Diversity by Task')
    plt.ylabel('Number of Unique Layer Types')
    plt.xlabel('Task')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'images', "layer_diversity_by_task.png"))
    plt.close()
    
    # Export to CSV
    complexity_df = pd.DataFrame({
        'Task': tasks,
        'Average Layers': avg_layers,
        'Layer Diversity': diversity
    })
    complexity_df.to_csv(os.path.join(OUTPUT_DIR, 'csv_data', 'model_complexity.csv'), index=False)

# Report Generation Functions
def create_architecture_patterns_report(task_dfs, hierarchy_patterns, specialized_components):
    """Generate a report on architecture patterns"""
    report_lines = []
    report_lines.append("# Neural Network Architecture Patterns")
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

def generate_task_summaries(task_dfs, model_dfs):
    """Generate summary reports for each task"""
    for task, df in task_dfs.items():
        summary_lines = []
        summary_lines.append(f"# {task} Architecture Summary")
        
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
        top10 = layer_counts.nlargest(10)
        for layer, count in top10.items():
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
    """Export comparative data for further analysis"""
    # Get common columns across all dataframes
    common_cols = set.intersection(*[set(df.columns) for df in task_dfs.values()])
    
    # For each task, summarize the data
    for task, df in task_dfs.items():
        # Summarize numeric columns
        numeric_summary = df.describe()
        numeric_summary.to_csv(os.path.join(OUTPUT_DIR, 'csv_data', f'{task}_numeric_summary.csv'))
        
        # Export layer type counts
        df['Layer Type'].value_counts().to_csv(
            os.path.join(OUTPUT_DIR, 'csv_data', f'{task}_layer_types.csv')
        )

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
        f.write("Top 10 layer types by task:\n\n")
        for task, counts in layer_type_counts.items():
            f.write(f"{task}:\n")
            top10 = counts.nlargest(10)
            for layer, count in top10.items():
                f.write(f"  {layer}: {count} ({count/sum(counts)*100:.1f}%)\n")
            f.write("\n")
    
    # Export top blocks by task
    with open(os.path.join(txt_dir, "top_blocks.txt"), "w") as f:
        f.write("Top 10 blocks by task:\n\n")
        for task, counts in block_counts.items():
            if not counts.empty:
                f.write(f"{task}:\n")
                top10 = counts.nlargest(10)
                for block, count in top10.items():
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
    print("Neural Network Architecture Analyzer")
    print("===================================")
    
    # Create output directories
    create_output_dirs()
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
    
    # Basic analysis
    print("\nPerforming basic analysis...")
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
    print("\nPerforming advanced analysis...")
    task_sequences = analyze_layer_sequences(model_dfs)
    common_blocks = extract_common_blocks(model_dfs)
    complexity = analyze_model_complexity(task_dfs)
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_layer_distributions(layer_type_counts)
    plot_task_comparison(layer_distribution_comparison)
    visualize_hierarchy(hierarchy_patterns)
    visualize_complexity(complexity)
    
    # Export data and statistics
    print("\nExporting data and statistics...")
    export_block_statistics(common_blocks)
    export_statistics_as_txt(layer_type_counts, block_counts, hierarchy_patterns, 
                           complexity, common_layers, unique_layers)
    export_comparative_data(task_dfs)
    
    # Generate reports
    print("\nGenerating reports...")
    create_architecture_patterns_report(task_dfs, hierarchy_patterns, specialized_components)
    generate_task_summaries(task_dfs, model_dfs)
    
    # Create index file
    create_index_file()
    
    # Print summary
    print("\nAnalysis complete!")
    print(f"Results saved to {OUTPUT_DIR}")
    print("- Visualizations: images/")
    print("- Data files: csv_data/")
    print("- Reports: reports/")
    print("- Statistics: statistics_txt/")
    print(f"- Index file: {os.path.join(OUTPUT_DIR, 'analysis_index.md')}")

if __name__ == "__main__":
    main() 
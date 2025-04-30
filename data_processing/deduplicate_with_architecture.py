"""
Deduplicate model architecture files by comparing actual layer structures
"""

import os
import pandas as pd
import hashlib
import shutil
from collections import defaultdict
import numpy as np
from difflib import SequenceMatcher

def calculate_similarity(df1, df2):
    """Calculate similarity between two model architectures"""
    # If number of layers differs too much, consider different architectures
    if abs(len(df1) - len(df2)) / max(len(df1), len(df2)) > 0.3:  # 30% threshold
        return 0.0
    
    # Compare layer types sequence
    layers1 = df1['Layer Type'].tolist()
    layers2 = df2['Layer Type'].tolist()
    
    # Use sequence matcher to get similarity ratio
    matcher = SequenceMatcher(None, layers1, layers2)
    layer_type_similarity = matcher.ratio()
    
    # Compare model structure (block organization)
    if 'Block' in df1.columns and 'Block' in df2.columns:
        blocks1 = df1['Block'].tolist()
        blocks2 = df2['Block'].tolist()
        # Get block counts
        block_count1 = len(set(blocks1))
        block_count2 = len(set(blocks2))
        
        # If block counts differ too much, reduce similarity
        block_similarity = 1.0 - abs(block_count1 - block_count2) / max(block_count1, block_count2)
    else:
        block_similarity = 1.0  # No block info available
    
    # Combine similarities (weight layer types more)
    combined_similarity = (0.7 * layer_type_similarity) + (0.3 * block_similarity)
    
    return combined_similarity

def get_architecture_hash(df):
    """Calculate a hash of the model architecture"""
    # Create a string representation of the architecture
    # Only focus on architecture, not specific model IDs or names
    layer_types = '|'.join(df['Layer Type'].tolist())
    shape_info = '|'.join([str(df['Input Shape'].iloc[0]), str(df['Output Shape'].iloc[-1])])
    if 'Block' in df.columns:
        blocks = '|'.join(sorted(df['Block'].unique()))
    else:
        blocks = ''
    
    # Combine layer types, shape info and blocks to create architecture string
    arch_str = f"{layer_types}|{shape_info}|{blocks}"
    return hashlib.md5(arch_str.encode()).hexdigest()

def deduplicate_models(similarity_threshold=0.85):
    """Deduplicate model architecture files based on architecture similarity"""
    # Define input and output directories
    input_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "model_architecture", "raw")
    output_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "model_architecture", "process_with_architecture")
    
    # Check if input directory exists
    if not os.path.exists(input_base_dir):
        print(f"Error: Input directory {input_base_dir} does not exist.")
        return
        
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Get all task directories
    task_dirs = [d for d in os.listdir(input_base_dir) 
                if os.path.isdir(os.path.join(input_base_dir, d))]
                
    if not task_dirs:
        print(f"No task directories found in {input_base_dir}")
        return

    # Process each task directory
    for task_dir in task_dirs:
        print(f"Processing task: {task_dir}")
        
        # Create output directory for this task
        output_dir = os.path.join(output_base_dir, task_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Input directory for this task
        input_dir = os.path.join(input_base_dir, task_dir)
        
        # Get all CSV files in the task directory
        csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"  No CSV files found in {task_dir}, skipping")
            continue
        
        print(f"  Found {len(csv_files)} CSV files")
        
        # Load all model architectures
        models = []
        for csv_file in csv_files:
            file_path = os.path.join(input_dir, csv_file)
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Add to models list
                models.append({
                    'file': csv_file,
                    'df': df,
                    'hash': get_architecture_hash(df)
                })
                
            except Exception as e:
                print(f"  Error processing {csv_file}: {e}")
        
        # Group similar architectures
        architecture_groups = []
        processed = set()
        
        # First pass: group by hash for exact matches
        hash_groups = defaultdict(list)
        for i, model in enumerate(models):
            hash_groups[model['hash']].append(i)
        
        # Create initial groups from hash matches
        for hash_val, indices in hash_groups.items():
            if len(indices) > 0:
                group = [models[i] for i in indices]
                architecture_groups.append(group)
                processed.update(indices)
        
        # Second pass: group by similarity for similar but not identical architectures
        for i, model1 in enumerate(models):
            if i in processed:
                continue
                
            # Create a new group with this model
            current_group = [model1]
            processed.add(i)
            
            # Check against all other unprocessed models
            for j, model2 in enumerate(models):
                if j in processed or i == j:
                    continue
                    
                # Calculate similarity
                similarity = calculate_similarity(model1['df'], model2['df'])
                
                # If similar enough, add to current group
                if similarity >= similarity_threshold:
                    current_group.append(model2)
                    processed.add(j)
            
            # Add the group to architecture groups
            architecture_groups.append(current_group)
        
        # Output deduplication results
        unique_architecture_count = len(architecture_groups)
        total_file_count = len(csv_files)
        print(f"  Found {unique_architecture_count} unique model architectures out of {total_file_count} files")
        
        # Create deduplication results CSV
        results = []
        
        # Process each architecture group
        for group_idx, group in enumerate(architecture_groups):
            # Choose the representative file - use first file as representative
            group.sort(key=lambda x: len(x['file']))  # Simple heuristic: choose shortest filename
            representative = group[0]
            
            # Copy the representative file to output directory
            shutil.copy(
                os.path.join(input_dir, representative['file']),
                os.path.join(output_dir, representative['file'])
            )
            
            # Record group information
            files_in_group = [model['file'] for model in group]
            hashes_in_group = [model['hash'] for model in group]
            
            # Extract base model name if available
            if 'Base Model Name' in representative['df'].columns:
                base_names = [name for name in representative['df']['Base Model Name'].unique() if str(name) != 'nan']
                base_model = base_names[0] if base_names else "unknown"
            else:
                base_model = os.path.splitext(representative['file'])[0]
                
            # Record results for this group
            results.append({
                'Architecture Group': f"arch_group_{group_idx}",
                'Base Model': base_model,
                'Representative File': representative['file'],
                'Architecture Hash': representative['hash'],
                'Group Size': len(group),
                'Files in Group': ', '.join(files_in_group),
                'Hashes in Group': ', '.join(hashes_in_group)
            })
        
        # Save deduplication results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'architecture_deduplication_results.csv'), index=False)
        print(f"  Saved deduplicated files and results to {output_dir}")

    print("Deduplication by architecture similarity completed!")

if __name__ == "__main__":
    deduplicate_models() 
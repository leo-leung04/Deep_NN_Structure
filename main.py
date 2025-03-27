#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main Script - Extract Hugging Face Model Architectures
Responsible for coordinating modules and executing the main workflow
"""

import os
import time
import logging
from tqdm import tqdm
import pandas as pd
import json
import re
import torch

# Import custom modules
from model_utils import create_model_for_task, extract_model_details
from api_utils import get_base_model, get_model_api_details
from task_utils import get_task_categories
from config import setup_logging, FALLBACK_TASKS

# Setup logging
logger = setup_logging()

def main():
    """Main execution function, coordinates the entire model extraction process"""
    # Create output directory
    base_output_dir = "huggingface_models"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Get task categories
    logger.info("Fetching task categories from Hugging Face...")
    task_categories = get_task_categories()
    
    if not task_categories:
        logger.warning("Failed to fetch task categories from website. Using predefined task list.")
        task_categories = FALLBACK_TASKS
    
    logger.info(f"Found {len(task_categories)} task categories")
    
    # Process each task category
    for task_id, task_info in task_categories.items():
        task_name = task_info['name']
        category_name = task_info['category']
        
        logger.info(f"Processing task: {task_name} (Category: {category_name})")
        
        # Create directory structure
        if category_name != "Unknown" and category_name != "Uncategorized":
            task_dir = os.path.join(base_output_dir, category_name, task_name)
        else:
            task_dir = os.path.join(base_output_dir, task_name)
            
        os.makedirs(task_dir, exist_ok=True)
        
        # Fetch models for this task
        try:
            # Try to list all models for this task
            from huggingface_hub import list_models
            models_for_task = list(list_models(filter=task_id))
            models_count = len(models_for_task)
            logger.info(f"Found {models_count} models for task {task_name}")
            
            if models_count == 0:
                logger.warning(f"No models found for task {task_name}. Skipping.")
                continue
            
            # Track unique base models for this task
            unique_base_models = {}
            
            # Limit number of models per task based on available resources
            max_models_per_task = min(models_count, 100)  # Adjust as needed
            
            # Iterate through models for this task
            for model_info in tqdm(models_for_task[:max_models_per_task], 
                                  desc=f"Processing {task_name} models"):
                # Get base model info
                base_model_name = get_base_model(model_info.modelId)
                
                if not base_model_name:
                    logger.warning(f"Could not determine base model for {model_info.modelId}. Skipping.")
                    continue
                
                # Skip if we already have a representative for this base model
                if base_model_name in unique_base_models:
                    continue
                
                # Save this model as the representative for this base model
                unique_base_models[base_model_name] = model_info.modelId
                
                # Create model and extract architecture
                logger.info(f"Loading model: {model_info.modelId}")
                model = create_model_for_task(model_info.modelId, task_id)
                
                if model is None:
                    logger.warning(f"Skipping {model_info.modelId} - Could not create model")
                    continue
                
                # Extract architecture details
                logger.info(f"Extracting architecture for: {model_info.modelId}")
                arch_details = extract_model_details(model, task_id, model_info.modelId)
                
                if arch_details is None:
                    logger.warning(f"Skipping {model_info.modelId} - Could not extract architecture")
                    # Free memory
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                
                # Get model metadata separately (not adding to each layer)
                metadata = {
                    "Model ID": model_info.modelId,
                    "Base Model Name": base_model_name,
                    "Task": task_name,
                    "Category": category_name
                }
                
                # Add API details to metadata
                api_details = get_model_api_details(model_info.modelId)
                metadata.update(api_details)
                
                # Convert complex data structures to strings
                df = pd.DataFrame(arch_details)
                df["Parameters"] = df["Parameters"].apply(lambda x: str(x))
                df["Input Shape"] = df["Input Shape"].apply(lambda x: str(x))
                df["Output Shape"] = df["Output Shape"].apply(lambda x: str(x))
                
                # Add metadata as a separate row at the end of the CSV
                # Create a new row with empty architecture columns, only metadata
                metadata_row = {col: "" for col in df.columns}
                metadata_row["Layer Name"] = "METADATA"
                
                # Add all metadata to this row
                for key, value in metadata.items():
                    # If metadata key not in DataFrame columns, add as new column
                    if key not in df.columns:
                        df[key] = ""
                    # Set value for metadata row
                    metadata_row[key] = value
                
                # Add metadata row to end of DataFrame
                df = pd.concat([df, pd.DataFrame([metadata_row])], ignore_index=True)
                
                # Create filename (sanitized)
                safe_base_name = re.sub(r'[^\w\-.]', '_', base_model_name)
                
                # Save detailed architecture (with metadata row at the end)
                csv_filename = f"{safe_base_name}_structure.csv"
                output_path = os.path.join(task_dir, csv_filename)
                df.to_csv(output_path, index=False)
                
                # Also save complete metadata as JSON (for programmatic access)
                metadata_filename = f"{safe_base_name}_metadata.json"
                metadata_path = os.path.join(task_dir, metadata_filename)
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Saved architecture of {model_info.modelId} to {output_path}")
                
                # Free memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Add small delay to avoid API rate limiting
                time.sleep(0.5)
                
            logger.info(f"Processed {len(unique_base_models)} unique base models for task {task_name}")
            
        except Exception as e:
            logger.error(f"Error processing task {task_name}: {e}")
            continue
    
    logger.info("Model extraction complete.")

if __name__ == "__main__":
    main() 
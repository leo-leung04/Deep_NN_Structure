"""
Main script for Hugging Face Model Architecture Extractor
"""

import os
import re
import time
import gc
from tqdm import tqdm
from huggingface_hub import list_models
import pandas as pd

# Import local modules
from .config import (
    OUTPUT_DIR, SLEEP_TIME, MIN_MODELS_THRESHOLD, 
    COMPLETED_TASKS, logger
)
from .utils import clear_all_model_files, get_all_tasks, get_base_model
from .model_loader import load_model_for_task
from .structure_extractor import extract_model_structure

def main():
    """Extract model architectures for all tasks"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Statistics
    total_models_found = 0
    total_processed = 0
    processed_base_models = set()
    models_since_last_cleanup = 0  # Track models processed since last cleanup
    
    # Get all tasks
    tasks = get_all_tasks()
    logger.info(f"Found {len(tasks)} tasks to process")
    
    # Load already processed base models (if resuming previous run)
    try:
        # Scan output directory to recover processed models list
        for task_name in COMPLETED_TASKS:
            task_dir = os.path.join(OUTPUT_DIR, task_name)
            if os.path.exists(task_dir):
                for filename in os.listdir(task_dir):
                    if filename.endswith("_structure.csv"):
                        base_name = filename.replace("_structure.csv", "")
                        processed_base_models.add(base_name)
                logger.info(f"Loaded {len(processed_base_models)} processed base models from existing outputs")
    except Exception as e:
        logger.warning(f"Error loading processed models: {e}")
    
    # Process each task
    for task in tasks:
        # Skip already completed tasks
        if task in COMPLETED_TASKS:
            logger.info(f"Skipping already completed task: {task}")
            continue
            
        logger.info(f"\n=== Processing Task: {task} ===")
        
        # Get PyTorch models for this task
        try:
            models = list(list_models(filter=task, library="pytorch"))
            model_count = len(models)
            logger.info(f"Found {model_count} PyTorch models for task '{task}'")
            total_models_found += model_count
            
            # Skip tasks with fewer models than threshold
            if model_count < MIN_MODELS_THRESHOLD:
                logger.info(f"Skipping task '{task}' - only {model_count} models (less than {MIN_MODELS_THRESHOLD})")
                continue
                
            # Create task directory
            task_dir = os.path.join(OUTPUT_DIR, task)
            os.makedirs(task_dir, exist_ok=True)
            
            # Track base models for this task
            task_base_models = {}
            task_processed = 0
            
            # Process each model
            for model_info in tqdm(models, desc=f"Processing {task} models"):
                model_id = model_info.modelId
                
                try:
                    # Get base model name
                    base_name = get_base_model(model_id)
                    safe_base_name = re.sub(r"[\W]+", "_", base_name)
                    
                    # Skip if already processed this base model
                    if safe_base_name in processed_base_models:
                        logger.info(f"Skipping {model_id} - Base model {base_name} already processed")
                        continue
                    
                    logger.info(f"Processing: {model_id} (base: {base_name})")
                    
                    # Load model
                    model = load_model_for_task(model_id, task)
                    if not model:
                        logger.warning(f"Failed to load model {model_id}")
                        continue
                    
                    # Extract structure
                    try:
                        df = extract_model_structure(model)
                        if df is None or df.empty:
                            logger.warning(f"Failed to extract structure for {model_id}")
                            continue
                            
                        # Add metadata
                        df["Model ID"] = model_id
                        df["Base Model Name"] = base_name
                        df["Task"] = task
                        
                        # Save to CSV
                        output_path = os.path.join(task_dir, f"{safe_base_name}_structure.csv")
                        df.to_csv(output_path, index=False)
                        logger.info(f"Saved structure to {output_path}")
                        
                        # Mark as processed
                        processed_base_models.add(safe_base_name)
                        task_base_models[safe_base_name] = model_id
                        task_processed += 1
                        total_processed += 1
                        models_since_last_cleanup += 1
                        
                        # Clean cache every 100 models
                        if models_since_last_cleanup >= 100:
                            logger.info("Cleaning up memory cache and model files after processing 100 models...")
                            clear_all_model_files()  # Clean all model files
                            models_since_last_cleanup = 0
                            
                    except Exception as e:
                        logger.error(f"Error extracting structure: {e}")
                except Exception as e:
                    logger.error(f"Error processing model: {e}")
                finally:
                    # Clean up memory
                    if 'model' in locals():
                        del model
                        gc.collect()
                    time.sleep(SLEEP_TIME)
            
            logger.info(f"Task '{task}': Processed {task_processed}/{len(models)} models, {len(task_base_models)} unique base models")
            
        except Exception as e:
            logger.error(f"Failed to process task '{task}': {e}")
    
    # Final statistics
    logger.info(f"Extraction complete! Found {total_models_found} models, extracted {len(processed_base_models)} unique base models")
    
    # Success summary
    if processed_base_models:
        logger.info("Processed base models:")
        for base_name in sorted(processed_base_models):
            logger.info(f"  - {base_name}")
    else:
        logger.warning("No base models were processed!")

if __name__ == "__main__":
    main() 
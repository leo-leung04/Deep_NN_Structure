"""
Utility functions for Hugging Face Model Architecture Extractor
"""

import os
import re
import requests
import shutil
from .config import logger

def clear_all_model_files():
    """Clean up all downloaded model files from cache"""
    try:
        # Get model cache path
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        # Check if cache directory exists
        if not os.path.exists(cache_dir):
            logger.info("No model cache directory found")
            return
            
        # Get all model directories in cache
        model_dirs = [d for d in os.listdir(cache_dir) 
                     if os.path.isdir(os.path.join(cache_dir, d))]
        
        if not model_dirs:
            logger.info("No model files found to delete")
            return
            
        # Delete all model directories
        deleted_count = 0
        for model_dir in model_dirs:
            try:
                model_path = os.path.join(cache_dir, model_dir)
                shutil.rmtree(model_path)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {model_dir}: {e}")
                
        logger.info(f"Deleted {deleted_count} model directories")
        
    except Exception as e:
        logger.warning(f"Failed to clear model files: {e}")

def get_all_tasks():
    """Retrieve all available tasks from Hugging Face"""
    tasks = []
    try:
        logger.info("Fetching tasks from Hugging Face API...")
        resp = requests.get("https://huggingface.co/api/tasks")
        resp.raise_for_status()
        tasks_data = resp.json()
        tasks = list(tasks_data.keys())
        logger.info(f"Found {len(tasks)} tasks via API")
    except Exception as e:
        logger.warning(f"API fetch failed: {e}, trying web scraping...")
        # Try web scraping as backup
        try:
            from bs4 import BeautifulSoup
            page = requests.get("https://huggingface.co/tasks")
            soup = BeautifulSoup(page.text, 'html.parser')
            links = soup.find_all('a', href=lambda x: x and x.startswith('/tasks/'))
            tasks = list({link['href'].split('/')[-1] for link in links})
            logger.info(f"Found {len(tasks)} tasks via web scraping")
        except Exception as e:
            logger.error(f"Web scraping failed: {e}")
    
    if not tasks:
        logger.error("No tasks could be retrieved.")
        raise RuntimeError("Failed to retrieve tasks from Hugging Face")
    
    return tasks

def get_base_model(model_id):
    """Extract base model name from model ID"""
    try:
        # Get model name from the model_id (e.g., "bert-base-uncased" -> "bert")
        parts = model_id.split('/')
        if len(parts) > 1:
            model_name = parts[-1]
        else:
            model_name = model_id
            
        # Extract first part before hyphen or underscore
        if '-' in model_name:
            base_name = model_name.split('-')[0]
        elif '_' in model_name:
            base_name = model_name.split('_')[0]
        else:
            base_name = model_name
            
        return base_name.lower()
    except Exception as e:
        logger.warning(f"Error extracting base model from ID {model_id}: {e}")
        return model_id.split('/')[-1].lower()  # Last resort 
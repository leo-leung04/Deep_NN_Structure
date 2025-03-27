#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API Utilities Module - Responsible for interacting with Hugging Face API to retrieve model metadata
"""

import re
import logging
import requests
from huggingface_hub import model_info

# Get logger
logger = logging.getLogger(__name__)

# -----------------------------
# Function: Get base model information
def get_base_model(model_id):
    """Get metadata from Hugging Face API to determine the base model"""
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Method 1: Check 'model_index' field
        if "model_index" in data and data["model_index"]:
            return data["model_index"][0]["name"]

        # Method 2: Check architecture or model type in 'config.json'
        config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
        response = requests.get(config_url, timeout=10)
        if response.status_code == 200:
            config = response.json()
            if "architectures" in config and config["architectures"]:
                return config["architectures"][0]  # Base architecture
            if "model_type" in config:
                return config["model_type"]

        # Method 3: Look for parent model
        if "pipeline_tag" in data and "parent_model" in data:
            return data["parent_model"]

        # Method 4: Try to get model_info
        try:
            m_info = model_info(model_id)
            if hasattr(m_info, 'cardData') and m_info.cardData and 'base_model' in m_info.cardData:
                return m_info.cardData['base_model']
        except:
            pass

        # Fallback: Use heuristic approach to extract from model_id
        model_name = model_id.split("/")[-1]
        # Remove common suffixes that don't represent architecture
        for suffix in ['-base', '-large', '-small', '-tiny', '-finetuned']:
            model_name = model_name.replace(suffix, '')
        
        # Extract possible architecture name
        arch_match = re.search(r'([a-zA-Z]+)(?:\d*|v\d+)', model_name)
        if arch_match:
            return arch_match.group(1).lower()
            
        return model_name

    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting base model info for {model_id}: {e}")
        return None

# -----------------------------
# Function: Get model API details
def get_model_api_details(model_id):
    """Get additional model details from Hugging Face API"""
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Extract key metadata
        model_details = {
            "Author": data.get("author", "Unknown"),
            "Tags": str(data.get("tags", [])),
            "Pipeline Tag": data.get("pipeline_tag", "Unknown"),
            "Downloads": data.get("downloads", 0),
            "Library": data.get("library_name", "Unknown"),
            "Last Modified": data.get("last_modified", "Unknown")
        }
        
        # Add more useful fields (if they exist)
        if "private" in data:
            model_details["Private"] = data["private"]
        if "likes" in data:
            model_details["Likes"] = data["likes"]
        if "sha" in data:
            model_details["SHA"] = data["sha"]
        if "siblings" in data:
            # Extract file list
            model_details["Files"] = str([s["rfilename"] for s in data["siblings"] if "rfilename" in s][:5])
            model_details["File Count"] = len(data["siblings"])
        
        return model_details
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting API details for {model_id}: {e}")
        return {} 
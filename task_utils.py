#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task Utilities Module - Responsible for retrieving Hugging Face task categories
"""

import logging
import requests
from bs4 import BeautifulSoup

# Get logger
logger = logging.getLogger(__name__)

# -----------------------------
# Function: Get all available task categories
def get_task_categories():
    """Retrieve all available task categories from the Hugging Face tasks page"""
    url = "https://huggingface.co/tasks"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all task categories and subcategories
        task_data = {}
        
        # Find all task category sections
        task_sections = soup.find_all('div', class_='lg:flex')
        
        for section in task_sections:
            # Try to find category heading
            category_heading = section.find('h3', class_='font-semibold')
            if not category_heading:
                continue
                
            category_name = category_heading.text.strip()
            
            # Find all task links in this category
            task_links = section.find_all('a', href=lambda href: href and '/tasks/' in href)
            
            for link in task_links:
                task_name = link.text.strip()
                task_id = link['href'].split('/')[-1]
                
                # Add to task data with category structure
                task_data[task_id] = {
                    'name': task_name,
                    'category': category_name,
                    'id': task_id
                }
        
        # If tasks were found through HTML parsing, return them
        if task_data:
            return task_data
            
        # Backup method: Try using API
        api_url = "https://huggingface.co/api/tasks"
        response = requests.get(api_url, timeout=15)
        if response.status_code == 200:
            api_tasks = response.json()
            # Format to our expected structure
            for task_id, task_info in api_tasks.items():
                category = task_info.get("category", "Uncategorized")
                name = task_info.get("name", task_id)
                task_data[task_id] = {
                    'name': name,
                    'category': category,
                    'id': task_id
                }
                
        return task_data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching task categories: {e}")
        return {} 
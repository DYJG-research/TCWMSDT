#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions module

Provide logging configuration plus checkpoint save/load helpers.
"""

import json
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure logging.
    
    Args:
        log_level: logging level
        log_file: optional log file path
    """
    # Create the log formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler when a log file path is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Reduce noise from common third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)

def save_checkpoint(data: Dict[str, Any], checkpoint_path: str):
    """
    Save checkpoint data.
    
    Args:
        data: payload to persist
        checkpoint_path: destination checkpoint path
    """
    try:
        # Append timestamp
        data["timestamp"] = datetime.now().isoformat()
        
        # Write to a temp file and rename for atomicity
        temp_path = checkpoint_path + ".tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Rename to final file
        os.rename(temp_path, checkpoint_path)
        
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")
        # Clean up the temp file if it still exists
        if os.path.exists(temp_path):
            os.remove(temp_path)

def load_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint data.
    
    Args:
        checkpoint_path: checkpoint file path
        
    Returns:
        Loaded checkpoint dict or None if unavailable
    """
    try:
        if not os.path.exists(checkpoint_path):
            return None
        
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.info(f"Successfully loaded checkpoint: {checkpoint_path}")
        return data
        
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        return None

def ensure_dir(path: str):
    """
    Ensure the directory exists.
    
    Args:
        path: directory path
    """
    os.makedirs(path, exist_ok=True)

def format_score(score: float, precision: int = 2) -> str:
    """
    Format score text.
    
    Args:
        score: score value
        precision: decimal places
        
    Returns:
        Formatted score string
    """
    return f"{score:.{precision}f}"

def format_percentage(score: float, precision: int = 1) -> str:
    """
    Format percentage text.
    
    Args:
        score: score between 0.0 and 1.0
        precision: decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{score * 100:.{precision}f}%"

def calculate_statistics(scores: list) -> Dict[str, float]:
    """
    Compute statistics for a score list.
    
    Args:
        scores: list of scores
        
    Returns:
        Dict with summary statistics
    """
    if not scores:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0
        }
    
    import numpy as np
    
    scores_array = np.array(scores)
    
    return {
        "mean": float(np.mean(scores_array)),
        "median": float(np.median(scores_array)),
        "std": float(np.std(scores_array)),
        "min": float(np.min(scores_array)),
        "max": float(np.max(scores_array)),
        "count": len(scores)
    }

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration data.
    
    Args:
        config: config dictionary
        
    Returns:
        True if valid, else False
    """
    required_fields = [
        "data_path",
        "reward_api_host",
        "reward_api_port",
        "llm_judge_api_host",
        "llm_judge_api_port",
        "reward_api_key",
        "llm_judge_api_key",
    ]
    
    for field in required_fields:
        if field not in config:
            logging.error(f"Missing required config field: {field}")
            return False
    
    # Validate the data path
    if not os.path.exists(config["data_path"]):
        logging.error(f"Data file not found: {config['data_path']}")
        return False
    
    return True

def create_default_config() -> Dict[str, Any]:
    """
    Create the default configuration dict.
    
    Returns:
        Default config dictionary
    """
    return {
        "data_path": "TCM-PIWEIBING.jsonl",
        "local_model_gpu_id": -1,
        "reward_api_host": "127.0.0.1",
        "reward_api_port": 8001,
        "reward_model_name": "Fangzheng-RM",
        "reward_api_key": "YOUR_REWARD_API_KEY",
        "llm_judge_api_host": "127.0.0.1", 
        "llm_judge_api_port": 8002,
        "llm_judge_model_name": "Qwen3-32B",
        "llm_judge_api_key": "YOUR_JUDGE_API_KEY",
        "random_seed": None,  # None means auto-generate a random seed
        "max_retries": 3,
        "checkpoint_interval": 10
    }

def merge_configs(default_config: Dict[str, Any], 
                 user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge default and user configs.
    
    Args:
        default_config: base config
        user_config: overriding config
        
    Returns:
        Combined config dictionary
    """
    merged_config = default_config.copy()
    
    for key, value in user_config.items():
        if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
            # Recursively merge nested dictionaries
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            merged_config[key] = value
    
    return merged_config

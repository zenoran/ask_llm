import os
import logging
from typing import List, Optional
import requests
import time


def get_available_models(base_url: str) -> List[str]:
    """
    Query Ollama API for available models.
    
    Args:
        base_url: The base URL of the Ollama API
        
    Returns:
        List of available model names, or empty list if API unreachable
    """
    models = []
    start_time = time.time()
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            if "models" in data:
                models = [model["name"] for model in data["models"]]
            print(f"⏱️ Ollama API query took {(time.time() - start_time) * 1000:.2f} ms, found {len(models)} models")
            return models
        else:
            print(f"⏱️ Ollama API query failed in {(time.time() - start_time) * 1000:.2f} ms: HTTP {response.status_code}")
    except Exception as e:
        print(f"⏱️ Ollama API query failed in {(time.time() - start_time) * 1000:.2f} ms: {str(e)}")
    finally:
        return models

def find_matching_model(model_name: str, available_models: List[str]) -> Optional[str]:
    """
    Find a model matching the provided name, supporting partial matches.
    
    Args:
        model_name: The name or partial name of the model to find
        available_models: List of available model names to search
        
    Returns:
        Full model name if found, None if no match
    """
    # Check for exact match first
    if model_name in available_models:
        return model_name
    
    # Check for partial matches
    matches = [m for m in available_models if model_name.lower() in m.lower()]
    
    # If we found exactly one match, return it
    if len(matches) == 1:
        return matches[0]
    
    # If we found multiple matches, return None (caller should handle this)
    return None

def init_models(
    ollama_url: str,
    cache_file_path: str,
    force_refresh: bool | None = None
) -> List[str]:
    """
    Initialize Ollama models from cache or API
    
    Args:
        ollama_url: URL for the Ollama API
        cache_file_path: Path to the cache file
        force_refresh: Whether to refresh models from API
        
    Returns:
        List of available Ollama models
    """
    cached_models = _read_models_from_cache(cache_file_path)
    
    # If not refreshing and cache exists, use cached models
    if not force_refresh and cached_models is not None:
        return cached_models
    
    # If refreshing or no cache exists, query API
    if force_refresh or cached_models is None:
        try:
            models = get_available_models(ollama_url)
            # Always write to cache, even if empty
            _write_models_to_cache(cache_file_path, models)
            return models
        except Exception as e:
            logging.warning(f"Error fetching Ollama models: {e}")
            # Write empty cache file if API fails
            _write_models_to_cache(cache_file_path, [])
            # Return cached models if they exist
            return cached_models or []
    
    return []

def _read_models_from_cache(cache_file_path: str) -> Optional[List[str]]:
    """Read available models from cache file if it exists"""
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'r') as f:
                models = [line.strip() for line in f.readlines() if line.strip()]
            logging.debug(f"Read {len(models)} models from cache")
            return models
        except Exception as e:
            logging.warning(f"Error reading models from cache: {e}")
    return None

def _write_models_to_cache(cache_file_path: str, models: List[str]) -> None:
    """Write models to the cache file"""
    try:
        with open(cache_file_path, 'w') as f:
            for model in models:
                f.write(f"{model}\n")
        logging.debug(f"Wrote {len(models)} models to cache")
    except Exception as e:
        logging.warning(f"Error writing models to cache: {e}")

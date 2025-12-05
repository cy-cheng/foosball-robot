import yaml
import os

def load_config(config_path):
    """
    Loads a YAML configuration file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_config_value(config, keys, default=None):
    """
    Safely retrieves a value from a nested dictionary given a list of keys.
    Returns default if any key in the path is not found.
    """
    current_value = config
    for key in keys:
        if isinstance(current_value, dict) and key in current_value:
            current_value = current_value[key]
        else:
            return default
    return current_value

"""
Phase 2 Configuration Loading

Loads configuration from YAML file for Phase 2 model.
"""

import yaml
from pathlib import Path

def load_phase2_config():
    """Load Phase 2 configuration from YAML file"""
    config_path = Path(__file__).parent / "phase2_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_model_config():
    """Get just the model configuration section"""
    config = load_phase2_config()
    return config['model']

def get_training_config():
    """Get just the training configuration section"""
    config = load_phase2_config()
    return config['training']
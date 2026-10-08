import yaml
import math
import os

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def calculate_niters_per_epoch(dataset_size, batch_size):
    niters_per_epoch = round(math.ceil(dataset_size / batch_size))
    return niters_per_epoch

def save_config_to_yaml(config, log_dir, filename="config.yaml"):
    os.makedirs(log_dir, exist_ok=True)
    yaml_file_path = os.path.join(log_dir, filename)
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False, allow_unicode=True)
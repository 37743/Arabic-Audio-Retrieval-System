import os
import configparser

def get_config():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root_dir, 'settings.cfg')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find settings.cfg at {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    return config

def save_config(config):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root_dir, 'settings.cfg')
    
    with open(config_path, 'w', encoding='utf-8') as configfile:
        config.write(configfile)
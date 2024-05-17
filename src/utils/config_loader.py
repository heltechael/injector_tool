import json
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self):
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as file:
            config = json.load(file)

        self._validate_config(config)
        return config

    def _validate_config(self, config):
        required_keys = [
            "USE_BEST_THUMBNAILS",
            "MAX_INJECTIONS_PER_IMAGE",
            'data_dir',
            'full_images_dir',
            'thumbnails_dir',
            'output_dir',
            'injected_full_images_dir',
        ]

        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise KeyError(f"Missing required keys in config file: {', '.join(missing_keys)}")

    def get(self, key):
        return self.config.get(key)
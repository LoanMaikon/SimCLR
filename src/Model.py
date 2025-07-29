from torchvision.transforms import v2
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np
import os
from time import strftime, localtime

class Model():
    def __init__(self, config_path, gpu_index, operation):
        self.operation = operation
        self.device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

        match operation:
            case "train_encoder":
                self._load_train_encoder_config(config_path)
                self._save_train_encoder_config() # Save on JSON

                self._load_backbone()
                self._fit_projection_head() # Make _remove_projection_head() too

                self._compute_normalization()
                self._save_normalization() # Save on JSON

                self._load_transform()
                self._load_dataloader()
        
       # ...

    def _load_transform(self):
        pass

    def _save_normalization(self):
        pass

    def _compute_normalization(self):
        pass

    def _load_dataloader(self):
        pass

    def _load_backbone(self):
        self.model = None

        match self.model_name:
            case 'resnet18':
                pass

            case 'resnet50':
                pass
            
            case 'mobilenet_v3_large':
                pass
            
            case 'mobilevit':
                pass
        
        assert self.model is not None, f"Model {self.model_name} doesn't exist."
    
    def _fit_projection_head(self):
        pass
    
    def _save_train_encoder_config(self):
        # Save the config so when testing, we can load it
        pass

    def _load_train_encoder_config(self, config_path):
        config = yaml.safe_load(open(config_path, 'r'))

        if not os.path.exists(config['output_path']):
            os.makedirs(config['output_path'])
        executions = os.listdir(config['output_path'])
        for i in range(1, 100):
            if f"execution_{i}" not in executions:
                config['output_path'] += f"/execution_{i}"
                break

        config['output_path'] += f'/{self.operation}' if not config['output_path'].endswith('/') else f'{self.operation}'
        config['datasets_path'] += '/' if not config['datasets_path'].endswith('/') else ''

        os.makedirs(config['output_path'], exist_ok=True)

        # Configs to class attributes
        self.output_path = str(config['output_path'])
        self.datasets_path = str(config['datasets_path'])
        self.batch_size = int(config['batch_size'])
        self.num_epochs = int(config['num_epochs'])
        self.num_workers = int(config['num_workers'])
        self.model_name = str(config['model'])
        self.transform_resize = tuple(config['transform_resize'])
        self.train_datasets = list(config['train_datasets'])
        self.val_percent = float(config['val_percent'])

        return config

    def write_on_log(self, text):
        time = strftime("%Y-%m-%d %H:%M:%S - ", localtime())
        mode = "w" if not os.path.exists(os.path.join(self.output_path, "log.txt")) else "a"

        with open(os.path.join(self.output_path, "log.txt"), mode) as file:
            file.write(time + text + "\n")

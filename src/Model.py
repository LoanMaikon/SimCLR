from torchvision.transforms import v2
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
import shutil
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np
import os
from time import strftime, localtime
import json

from .resnet50 import resnet50
from .custom_dataset import custom_dataset
from .nt_xent import nt_xent

class Model():
    def __init__(self, config_path, gpu_index, operation):
        self.operation = operation
        self.device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

        match operation:
            case "train_encoder":
                self._create_train_encoder_output_path(config_path)
                self._load_train_encoder_config(config_path)

                self._load_backbone()
                self._fit_projection_head()

                self._compute_normalization() # Calculating mean and std
                self._save_normalization() # Save on JSON

                self._load_train_encoder_transform()
                self._load_train_encoder_dataloaders()

                self._load_train_encoder_criterion()

                self._load_train_encoder_optimizer()
                self._load_train_encoder_scheduler()

       # ...

    def get_learning_rate(self):
        return float(self.get_optimizer().param_groups[0]['lr'])

    def get_batch_size(self):
        return self.train_encoder_batch_size

    def get_target_batch_size(self):
        return self.train_encoder_target_batch_size

    def get_criterion(self):
        return self.criterion
    
    def apply_criterion(self, z1, z2):
        return self.criterion(z1, z2)

    def get_scheduler(self):
        return self.scheduler
    
    def get_optimizer(self):
        return self.optimizer
    
    def _load_train_encoder_criterion(self):
        self.criterion = nt_xent(self.train_encoder_temperature)

    def _load_train_encoder_scheduler(self):
        def __lr_lambda(current_epoch):
            if current_epoch < self.train_encoder_warmup_epochs:
                return float(current_epoch) / float(max(1, self.train_encoder_warmup_epochs))
            
            progress = (current_epoch - self.train_encoder_warmup_epochs) / (self.train_encoder_num_epochs - self.train_encoder_warmup_epochs)
            return 0.5 * (1. + np.cos(np.pi * progress)) # Cosine decay formula

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=__lr_lambda)

    '''
    The SimCLR defines lr = 0.3 * batch_size / 256
    '''
    def _load_train_encoder_optimizer(self):
        base_optimizer = optim.Adam(self.model.parameters(), lr=0.3 * self.train_encoder_batch_size / 256, weight_decay=1e-6)
        self.optimizer = base_optimizer

        # Still have to make LARS

    def save_model(self):
        os.makedirs(self.train_encoder_output_path + "/models", exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(self.train_encoder_output_path, "models", "model.pth"))

    def get_train_dataloader(self):
        return self.train_dataloader
    
    def get_val_dataloader(self):
        return self.val_dataloader

    def model_infer(self, x1, x2=None):
        x1 = x1.to(self.device)

        if x2 is not None:
            x2 = x2.to(self.device)

            return self.model(x1, x2)
        return self.model(x1)

    def get_train_encoder_num_epochs(self):
        return self.train_encoder_num_epochs

    def model_to_train(self):
        self.model.train()

    def model_to_eval(self):
        self.model.eval()

    def _load_train_encoder_dataloaders(self):
        train_dataset = custom_dataset(
            operation="train",
            apply_data_augmentation=True,
            datasets=self.train_encoder_train_datasets,
            datasets_folder_path=self.train_encoder_datasets_path,
            transform=self.transform
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.train_encoder_batch_size,
            num_workers=self.train_encoder_num_workers,
            shuffle=True,
            pin_memory=True,
        )

        val_dataset = custom_dataset(
            operation="val",
            apply_data_augmentation=True,
            datasets=self.train_encoder_train_datasets,
            datasets_folder_path=self.train_encoder_datasets_path,
            transform=self.transform
        )

        self.val_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=self.train_encoder_batch_size,
            num_workers=self.train_encoder_num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def _load_train_encoder_transform(self):
        # Pseudo code of Apendix A from SimCLR paper
        def __get_color_distortion(strength=1.0):
            collor_jitter = v2.ColorJitter(0.8 * strength, 0.8 * strength, 0.8 * strength, 0.2 * strength)
            rnd_color_jitter = v2.RandomApply([collor_jitter], p=0.8)
            rnd_gray = v2.RandomGrayscale(p=0.2)

            return v2.Compose([rnd_color_jitter, rnd_gray])

        self.transform = v2.Compose([
            v2.RandomResizedCrop(self.train_encoder_transform_resize), # SimCLR uses default scale and ratio
            v2.RandomHorizontalFlip(0.5), # SimCLR uses 50% probability
            __get_color_distortion(),
            v2.GaussianBlur(kernel_size=23), # SimCLR uses kernel size of 10% of image size and default sigma. We will use 23
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.mean, std=self.std)
        ])

    def _save_normalization(self):
        with open(os.path.join(self.train_encoder_output_path, "normalization.json"), "w") as f:
            json.dump({"mean": self.mean, "std": self.std}, f)

    def _compute_normalization(self):
        num_channels = 3

        transform = v2.Compose([
            v2.Resize(self.train_encoder_transform_resize),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        dataset = custom_dataset(
            operation="train",
            apply_data_augmentation=False,
            datasets=self.train_encoder_train_datasets,
            datasets_folder_path=self.train_encoder_datasets_path,
            transform=transform
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.train_encoder_batch_size,
            num_workers=self.train_encoder_num_workers,
            pin_memory=True,
        )

        mean = torch.zeros(num_channels, device=self.device)
        std = torch.zeros(num_channels, device=self.device)
        total_samples = 0
        for images, _ in dataloader:
            images = images.to(self.device)
            batch_samples = images.size(0)
            total_samples += batch_samples
            for i in range(num_channels):
                mean[i] += images[:, i, :, :].mean().item() * batch_samples
                std[i] += images[:, i, :, :].std().item() * batch_samples

        self.mean = [m.cpu().item() / total_samples for m in mean]
        self.std = [s.cpu().item() / total_samples for s in std]

    def _load_backbone(self):
        self.model = None

        match self.train_encoder_model_name:
            case 'resnet18':
                pass

            case 'resnet50':
                self.model = resnet50(self.train_encoder_projection_head_mode, self.train_encoder_projection_dim)
            
            case 'mobilenet_v3_large':
                pass
            
            case 'mobilevit':
                pass
        
        assert self.model is not None, f"Model {self.train_encoder_model_name} doesn't exist."

        self.model.to(self.device)
    
    def _fit_projection_head(self):
        self.model.fit_projection_head()
    
    def _remove_projection_head(self):
        self.model.remove_projection_head()

    def _create_train_encoder_output_path(self, config_path):
        config = yaml.safe_load(open(config_path, 'r'))

        if not os.path.exists(config['output_path']):
            os.makedirs(config['output_path'])
        executions = os.listdir(config['output_path'])
        for i in range(1, 100):
            if f"execution_{i}" not in executions:
                config['output_path'] += f"/execution_{i}"
                break
        config['output_path'] += f'/{self.operation}' if not config['output_path'].endswith('/') else f'{self.operation}'
        self.train_encoder_output_path = str(config['output_path'])

        os.makedirs(self.train_encoder_output_path, exist_ok=True)
        shutil.copyfile(config_path, os.path.join(self.train_encoder_output_path, "config.yaml"))

    def _load_train_encoder_config(self, config_path):
        config = yaml.safe_load(open(config_path, 'r'))

        config['output_path'] += '/' if not config['output_path'].endswith('/') else ''
        config['datasets_path'] += '/' if not config['datasets_path'].endswith('/') else ''

        # Configs to class attributes
        self.train_encoder_datasets_path = str(config['datasets_path'])
        self.train_encoder_batch_size = int(config['batch_size'])
        self.train_encoder_num_epochs = int(config['num_epochs'])
        self.train_encoder_num_workers = int(config['num_workers'])
        self.train_encoder_model_name = str(config['model'])
        self.train_encoder_transform_resize = tuple(config['transform_resize'])
        self.train_encoder_train_datasets = list(config['train_datasets'])
        self.train_encoder_projection_head_mode = str(config['projection_head_mode'])
        self.train_encoder_temperature = float(config['temperature'])
        self.train_encoder_projection_dim = int(config['projection_dim'])
        self.train_encoder_warmup_epochs = int(config['warmup_epochs'])
        self.train_encoder_target_batch_size = int(config['target_batch_size'])

    def write_on_log(self, text):
        time = strftime("%Y-%m-%d %H:%M:%S - ", localtime())
        mode = "w" if not os.path.exists(os.path.join(self.train_encoder_output_path, "log.txt")) else "a"

        with open(os.path.join(self.train_encoder_output_path, "log.txt"), mode) as file:
            file.write(time + text + "\n")

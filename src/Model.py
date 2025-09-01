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

from .resnet18 import resnet18
from .resnet50 import resnet50
from .custom_dataset import custom_dataset
from .nt_xent import nt_xent
from .lars import LARS

NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'imagenet': 1000,
    'dtd': 47,
    'fgvc-aircraft': 100,
    'flowers-102': 102,
    'food-101': 101,
    'oxford-pets': 37,
    'stanford-cars': 196,
    'tiny-imagenet': 200,
    'caltech-101': 101,
}

class Model():
    def __init__(self,
                config_path,
                gpu_index, operation,
                execution_name=None,
                label_fraction=None,
                lr=None,
                weight_decay=None,
                num_epochs=None,
                encoder_config=None,
                ):

        self.operation = operation
        self.execution_name = execution_name if execution_name is not None else None
        self.device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu') if gpu_index is not None else torch.device('cpu')

        # torch.set_num_threads(os.cpu_count())
        # torch.set_num_interop_threads(os.cpu_count())

        match operation:
            case "train_encoder":
                self._create_train_encoder_output_path(config_path)
                self._load_train_encoder_config(config_path)

                self._load_backbone()
                self._fit_projection_head()

                self.model.to(self.device)

                self._compute_normalization() # Calculating mean and std
                self._save_normalization() # Save on JSON

                self._load_train_encoder_transform()
                self._load_train_encoder_dataloaders()

                self._load_train_encoder_criterion()
                self._load_train_encoder_optimizer()
                self._load_train_encoder_scheduler()
            
            case "linear_evaluation":
                self.linear_evaluation_num_epochs = num_epochs
                self.linear_evaluation_label_fraction = label_fraction
                self.linear_evaluation_lr = lr
                self.linear_evaluation_weight_decay = weight_decay
                assert [label_fraction, num_epochs, lr, weight_decay, encoder_config, execution_name] is not None, "Label fraction, number of epochs, learning rate and weight decay must be provided for linear evaluation."

                self._load_linear_evaluation_config(config_path)
                self._load_train_encoder_config(encoder_config)

                self._create_linear_evaluation_output_path(encoder_config)

                self._load_backbone()
                self._load_encoder_weight()
                self._fit_classifier_head()
                self._freeze_encoder()

                self.model.to(self.device)

                self._load_normalization() # Load mean and std from JSON

                self._load_linear_evaluation_transform()
                self._load_linear_evaluation_dataloaders()

                self._load_linear_evaluation_criterion()
                self._load_linear_evaluation_optimizer()

            case "transfer_learning":
                self.transfer_learning_num_epochs = num_epochs
                self.transfer_learning_label_fraction = label_fraction
                self.transfer_learning_lr = lr
                self.transfer_learning_weight_decay = weight_decay
                assert [label_fraction, num_epochs, lr, weight_decay, encoder_config, execution_name] is not None, "Label fraction, number of epochs, learning rate and weight decay must be provided for transfer learning."

                self._load_transfer_learning_config(config_path)
                self._load_train_encoder_config(encoder_config)

                self._create_transfer_learning_output_path(encoder_config)

                self._load_backbone()
                self._load_encoder_weight()
                self._fit_classifier_head()
                self._unfreeze_encoder()

                self.model.to(self.device)

                self._load_normalization() # Load mean and std from JSON

                self._load_transfer_learning_transform()
                self._load_transfer_learning_dataloaders()

                self._load_transfer_learning_criterion()
                self._load_transfer_learning_optimizer()

    def get_transfer_learning_num_epochs(self):
        return self.transfer_learning_num_epochs

    def get_transfer_learning_batch_size(self):
        return self.transfer_learning_batch_size

    def get_linear_evaluation_train_datasets(self):
        return self.linear_evaluation_train_datasets
    
    def get_linear_evaluation_batch_size(self):
        return self.linear_evaluation_batch_size
    
    def get_linear_evaluation_num_epochs(self):
        return self.linear_evaluation_num_epochs

    def get_learning_rate(self):
        return float(self.get_optimizer().param_groups[0]['lr'])

    def get_train_encoder_batch_size(self):
        return self.train_encoder_batch_size
    
    def get_gpu_index(self):
        return self.device.index if self.device.type == 'cuda' else None

    def get_criterion(self):
        return self.criterion
    
    def get_use_val_subset(self):
        use_val_subset = None

        match self.operation:
            case "train_encoder":
                use_val_subset = self.train_encoder_use_val_subset
            case "linear_evaluation":
                use_val_subset = self.linear_evaluation_use_val_subset
            case "transfer_learning":
                use_val_subset = self.transfer_learning_use_val_subset
        
        return use_val_subset
    
    '''
    Check if the dataset has a validation set
    We use the validation set of imagenet to test
    '''
    def has_validation_set(self):
        use_val_subset = None
        set = None

        match self.operation:
            case "train_encoder":
                set = self.train_encoder_train_datasets
                use_val_subset = self.train_encoder_use_val_subset
            case "linear_evaluation":
                set = self.linear_evaluation_train_datasets
                use_val_subset = self.linear_evaluation_use_val_subset
            case "transfer_learning":
                set = self.transfer_learning_train_datasets
                use_val_subset = self.transfer_learning_use_val_subset
        
        if use_val_subset is not None and use_val_subset:
            return True
        
        for s in set:
            if s in ['imagenet', 'tiny-imagenet', 'caltech-101', 'cifar10', 'cifar100', 'food-101', 'stanford-cars', 'oxford-pets']:
                return False
        return True

    
    def apply_criterion(self, z1, z2):
        return self.criterion(z1, z2)

    def get_scheduler(self):
        return self.scheduler
    
    def get_optimizer(self):
        return self.optimizer
    
    def _load_transfer_learning_criterion(self):
        self.criterion = nn.CrossEntropyLoss()
    
    def _load_linear_evaluation_criterion(self):
        self.criterion = nn.CrossEntropyLoss()
    
    def _load_train_encoder_criterion(self):
        self.criterion = nt_xent(self.train_encoder_temperature)

    def _load_train_encoder_scheduler(self):
        def __lr_lambda(current_epoch):
            if current_epoch < self.train_encoder_warmup_epochs:
                return float(current_epoch) / float(max(1, self.train_encoder_warmup_epochs))
            
            progress = (current_epoch - self.train_encoder_warmup_epochs) / (self.train_encoder_num_epochs - self.train_encoder_warmup_epochs)
            return 0.5 * (1. + np.cos(np.pi * progress)) # Cosine decay formula

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=__lr_lambda)

    def _load_transfer_learning_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.transfer_learning_lr, momentum=0.9, weight_decay=self.transfer_learning_weight_decay, nesterov=True)

    def _load_linear_evaluation_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.linear_evaluation_lr, momentum=0.9, weight_decay=self.linear_evaluation_weight_decay, nesterov=True)

    def _load_train_encoder_optimizer(self):
        base_optimizer = optim.SGD(self.model.parameters(), lr=self.train_encoder_lr, momentum=0.9, weight_decay=self.train_encoder_weight_decay)
        self.optimizer = LARS(base_optimizer, trust_coefficient=0.001)

    def save_model(self):
        match self.operation:
            case "train_encoder":
                os.makedirs(self.train_encoder_output_path + "/models", exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.train_encoder_output_path, "models", "model.pth"))

            case "linear_evaluation":
                models_path = self.linear_evaluation_output_path + "models"
                os.makedirs(models_path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(models_path, f"model_lf_{str(self.linear_evaluation_label_fraction)}_ne_{str(self.linear_evaluation_num_epochs)}_lr_{str(self.linear_evaluation_lr)}_wd_{str(self.linear_evaluation_weight_decay)}.pth"))

            case "transfer_learning":
                models_path = self.transfer_learning_output_path + "/models"
                os.makedirs(models_path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(models_path, f"model_lf_{str(self.transfer_learning_label_fraction)}_ne_{str(self.transfer_learning_num_epochs)}_lr_{str(self.transfer_learning_lr)}_wd_{str(self.transfer_learning_weight_decay)}.pth"))

    def get_train_dataloader(self):
        return self.train_dataloader
    
    def get_val_dataloader(self):
        return self.val_dataloader
    
    def get_test_dataloader(self):
        return self.test_dataloader

    def model_infer(self, x1, x2=None):
        if x2 is None:
            x1 = self.model(x1.to(self.device))
            return x1

        else:
            x1, x2 = self.model(x1.to(self.device), x2.to(self.device))

            return x1, x2

    def get_train_encoder_num_epochs(self):
        return self.train_encoder_num_epochs

    def model_to_train(self):
        self.model.train()

    def model_to_eval(self):
        self.model.eval()

    def _load_transfer_learning_dataloaders(self):
        train_dataset = custom_dataset(
            operation="train",
            apply_data_augmentation=False,
            datasets=self.transfer_learning_train_datasets,
            datasets_folder_path=self.train_encoder_datasets_path,
            transform=self.transfer_learning_transform_train,
            label_fraction=self.transfer_learning_label_fraction,
            use_val_subset=self.transfer_learning_use_val_subset,
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.transfer_learning_batch_size,
            num_workers=self.transfer_learning_num_workers,
            shuffle=True,
            pin_memory=True if self.transfer_learning_pin_memory and self.device.type == 'cuda' else False,
            prefetch_factor=self.transfer_learning_prefetch_factor,
        )

        self.val_dataloader = None
        if self.transfer_learning_use_val_subset:
            val_dataset = custom_dataset(
                operation="val",
                apply_data_augmentation=False,
                datasets=self.transfer_learning_train_datasets,
                datasets_folder_path=self.train_encoder_datasets_path,
                transform=self.transfer_learning_transform_val_test,
                use_val_subset=self.transfer_learning_use_val_subset,
            )

            self.val_dataloader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=self.transfer_learning_batch_size,
                num_workers=self.transfer_learning_num_workers,
                shuffle=False,
                pin_memory=True if self.transfer_learning_pin_memory and self.device.type == 'cuda' else False,
                prefetch_factor=self.transfer_learning_prefetch_factor,
            )
        
        test_dataset = custom_dataset(
            operation="test",
            apply_data_augmentation=False,
            datasets=self.transfer_learning_train_datasets,
            datasets_folder_path=self.train_encoder_datasets_path,
            transform=self.transfer_learning_transform_val_test,
            use_val_subset=self.transfer_learning_use_val_subset,
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.transfer_learning_batch_size,
            num_workers=self.transfer_learning_num_workers,
            shuffle=False,
            pin_memory=True if self.transfer_learning_pin_memory and self.device.type == 'cuda' else False,
            prefetch_factor=self.transfer_learning_prefetch_factor,
        )

    def _load_linear_evaluation_dataloaders(self):
        train_dataset = custom_dataset(
            operation="train",
            apply_data_augmentation=False,
            datasets=self.linear_evaluation_train_datasets,
            datasets_folder_path=self.train_encoder_datasets_path,
            transform=self.linear_evaluation_transform_train,
            label_fraction=self.linear_evaluation_label_fraction,
            use_val_subset=self.linear_evaluation_use_val_subset,
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.linear_evaluation_batch_size,
            num_workers=self.linear_evaluation_num_workers,
            shuffle=True,
            pin_memory=True if self.linear_evaluation_pin_memory and self.device.type == 'cuda' else False,
            prefetch_factor=self.linear_evaluation_prefetch_factor,
        )

        self.val_dataloader = None
        if self.linear_evaluation_use_val_subset:
            val_dataset = custom_dataset(
                operation="val",
                apply_data_augmentation=False,
                datasets=self.linear_evaluation_train_datasets,
                datasets_folder_path=self.train_encoder_datasets_path,
                transform=self.linear_evaluation_transform_val_test,
                use_val_subset=self.linear_evaluation_use_val_subset,
            )

            self.val_dataloader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=self.linear_evaluation_batch_size,
                num_workers=self.linear_evaluation_num_workers,
                shuffle=False,
                pin_memory=True if self.linear_evaluation_pin_memory and self.device.type == 'cuda' else False,
                prefetch_factor=self.linear_evaluation_prefetch_factor,
            )

        test_dataset = custom_dataset(
            operation="test",
            apply_data_augmentation=False,
            datasets=self.linear_evaluation_train_datasets,
            datasets_folder_path=self.train_encoder_datasets_path,
            transform=self.linear_evaluation_transform_val_test,
            use_val_subset=self.linear_evaluation_use_val_subset,
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.linear_evaluation_batch_size,
            num_workers=self.linear_evaluation_num_workers,
            shuffle=False,
            pin_memory=True if self.linear_evaluation_pin_memory and self.device.type == 'cuda' else False,
            prefetch_factor=self.linear_evaluation_prefetch_factor,
        )

    def _load_train_encoder_dataloaders(self):
        train_dataset = custom_dataset(
            operation="train",
            apply_data_augmentation=True,
            datasets=self.train_encoder_train_datasets,
            datasets_folder_path=self.train_encoder_datasets_path,
            transform=self.transform,
            use_val_subset= self.train_encoder_use_val_subset,
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.train_encoder_batch_size,
            num_workers=self.train_encoder_num_workers,
            shuffle=True,
            pin_memory=True if self.train_encoder_pin_memory and self.device.type == 'cuda' else False,
            prefetch_factor=self.train_encoder_prefetch_factor,
        )

        if self.train_encoder_use_val_subset:
            val_dataset = custom_dataset(
                operation="val",
                apply_data_augmentation=True,
                datasets=self.train_encoder_train_datasets,
                datasets_folder_path=self.train_encoder_datasets_path,
                transform=self.transform,
                use_val_subset=self.train_encoder_use_val_subset
            )

            self.val_dataloader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=self.train_encoder_batch_size,
                num_workers=self.train_encoder_num_workers,
                shuffle=False,
                pin_memory=True if self.train_encoder_pin_memory and self.device.type == 'cuda' else False,
                prefetch_factor=self.train_encoder_prefetch_factor,
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

    def _load_transfer_learning_transform(self):
        self.transfer_learning_transform_train = v2.Compose([
            v2.RandomResizedCrop(self.train_encoder_transform_resize), # SimCLR B.5.
            v2.RandomHorizontalFlip(0.5), # SimCLR B.5.
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.mean, std=self.std),
        ])

        self.transfer_learning_transform_val_test = v2.Compose([
            v2.Resize(self.train_encoder_transform_resize),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.mean, std=self.std),
        ])

    def _load_linear_evaluation_transform(self):
        self.linear_evaluation_transform_train = v2.Compose([
            v2.RandomResizedCrop(self.train_encoder_transform_resize), # SimCLR B.6.
            v2.RandomHorizontalFlip(0.5) , # SimCLR B.6.
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.mean, std=self.std)
        ])

        self.linear_evaluation_transform_val_test = v2.Compose([
            v2.Resize(self.train_encoder_transform_resize),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.mean, std=self.std)
        ])

    def save_results(self, targets, all_predictions, loss, json_name):
        def _compute_top_accuracy(all_predictions, targets, top_k=1):
            correct = 0
            total = 0

            for i in range(len(targets)):
                top_k_indices = np.argsort(all_predictions[i])[-top_k:]

                if targets[i] in top_k_indices:
                    correct += 1
                total += 1
            
            return correct / total if total > 0 else 0.0

        def _compute_mean_per_class_accuracy(all_predictions, targets):
            class_correct = {}
            class_total = {}

            for i in range(len(targets)):
                label = targets[i]
                pred = all_predictions[i].argmax()

                if label not in class_correct:
                    class_correct[label] = 0
                    class_total[label] = 0
                
                if pred == label:
                    class_correct[label] += 1
                class_total[label] += 1

            return sum(class_correct[label] / class_total[label] for label in class_correct) / len(class_correct)

        output_path = None
        label_fraction = None
        match self.operation:
            case "linear_evaluation": output_path = self.linear_evaluation_output_path
            case "transfer_learning": output_path = self.transfer_learning_output_path
            case _: assert False, f"Operation {self.operation} does not save results."

        top_5_accuracy = _compute_top_accuracy(all_predictions, targets, top_k=5)
        top_1_accuracy = _compute_top_accuracy(all_predictions, targets, top_k=1)
        mean_per_class_accuracy = _compute_mean_per_class_accuracy(all_predictions, targets)

        with open(os.path.join(output_path, json_name), "w") as f:
            json.dump({
                "top_5_accuracy": top_5_accuracy,
                "top_1_accuracy": top_1_accuracy,
                "mean_per_class_accuracy": mean_per_class_accuracy,
                "loss": loss,
            }, f, indent=4)

    def _load_normalization(self):
        normalization_json_path = f"{self.train_encoder_output_path}{self.execution_name}/train_encoder/normalization.json"
        normalization_json = json.load(open(normalization_json_path, "r"))

        self.mean = normalization_json["mean"]
        self.std = normalization_json["std"]

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
            pin_memory=True if self.train_encoder_pin_memory and self.device.type == 'cuda' else False,
        )

        mean = torch.zeros(num_channels, device=self.device)
        std = torch.zeros(num_channels, device=self.device)
        total_pixels = 0

        for images, _ in dataloader:
            images = images.to(self.device)
            batch_samples = images.size(0)
            pixels_per_channel = batch_samples * images.size(2) * images.size(3)
            total_pixels += pixels_per_channel

            mean += images.sum(dim=[0, 2, 3])
            std += (images ** 2).sum(dim=[0, 2, 3])

        mean /= total_pixels
        std = torch.sqrt(std / total_pixels - mean ** 2)

        self.mean = mean.cpu().tolist()
        self.std = std.cpu().tolist()

    def get_device(self):
        return self.device

    def _load_weight(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist.")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

    def _load_encoder_weight(self):
        model_path = f"{self.train_encoder_output_path}{self.execution_name}/train_encoder/models/model.pth"

        self._fit_projection_head()
        self._load_weight(model_path)
        self._remove_projection_head()

    def _load_backbone(self):
        self.model = None
        use_checkpoint = None
        match self.operation:
            case "train_encoder": use_checkpoint = self.train_encoder_use_checkpoint
            case "linear_evaluation": use_checkpoint = self.linear_evaluation_use_checkpoint
            case "transfer_learning": use_checkpoint = self.transfer_learning_use_checkpoint
        assert use_checkpoint is not None, "Use checkpoint must be defined for the operation."

        match self.train_encoder_model_name:
            case 'resnet18':
                self.model = resnet18(self.train_encoder_projection_head_mode, self.train_encoder_projection_dim, use_checkpoint, self.train_encoder_pretrained)

            case 'resnet50':
                self.model = resnet50(self.train_encoder_projection_head_mode, self.train_encoder_projection_dim, use_checkpoint, self.train_encoder_pretrained)
            
            case 'mobilenet_v3_large':
                pass
            
            case 'mobilevit':
                pass
        
        assert self.model is not None, f"Model {self.train_encoder_model_name} doesn't exist."

        self.model.to(self.device)
    
    def _freeze_encoder(self):
        self.model.freeze_encoder()

    def _unfreeze_encoder(self):
        self.model.unfreeze_encoder()

    def _fit_classifier_head(self):
        train_dataset = None
        match self.operation:
            case "train_encoder": train_dataset = self.train_encoder_train_datasets[0]
            case "linear_evaluation": train_dataset = self.linear_evaluation_train_datasets[0]
            case "transfer_learning": train_dataset = self.transfer_learning_train_datasets[0]

        self.model.fit_classifier_head(num_classes=NUM_CLASSES[train_dataset])

    def _fit_projection_head(self):
        self.model.fit_projection_head()
    
    def _remove_projection_head(self):
        self.model.remove_projection_head()

    def _create_transfer_learning_output_path(self, encoder_config):
        datasets_str = "_".join(self.transfer_learning_train_datasets)

        self.transfer_learning_output_path = self.train_encoder_output_path + f"{self.execution_name}/transfer_learning_{datasets_str}/lf_{self.transfer_learning_label_fraction}_ne_{self.transfer_learning_num_epochs}_lr_{self.transfer_learning_lr}_wd_{self.transfer_learning_weight_decay}/"

        os.makedirs(self.transfer_learning_output_path, exist_ok=True)
        shutil.copyfile(encoder_config, os.path.join(self.transfer_learning_output_path, "encoder_config.yaml"))
        shutil.copyfile(self.transfer_learning_config_path, os.path.join(self.transfer_learning_output_path, "config.yaml"))

    def _create_linear_evaluation_output_path(self, encoder_config):
        datasets_str = "_".join(self.linear_evaluation_train_datasets)

        self.linear_evaluation_output_path = self.train_encoder_output_path + f"{self.execution_name}/linear_evaluation_{datasets_str}/lf_{self.linear_evaluation_label_fraction}_ne_{self.linear_evaluation_num_epochs}_lr_{self.linear_evaluation_lr}_wd_{self.linear_evaluation_weight_decay}"
        self.linear_evaluation_output_path += "_val_subset/" if self.linear_evaluation_use_val_subset else "/"

        os.makedirs(self.linear_evaluation_output_path, exist_ok=True)
        shutil.copyfile(encoder_config, os.path.join(self.linear_evaluation_output_path, "encoder_config.yaml"))
        shutil.copyfile(self.linear_evaluation_config_path, os.path.join(self.linear_evaluation_output_path, "config.yaml"))

    def _create_train_encoder_output_path(self, config_path):
        config = yaml.safe_load(open(config_path, 'r'))

        if not os.path.exists(config['output_path']):
            os.makedirs(config['output_path'])
        executions = os.listdir(config['output_path'])
        for i in range(1, 100):
            if f"execution_{i}" not in executions:
                config['output_path'] += f"/execution_{i}"
                break
        config['output_path'] += f'/{self.operation}/' if not config['output_path'].endswith('/') else f'/{self.operation}/'
        self.train_encoder_output_path = str(config['output_path'])

        os.makedirs(self.train_encoder_output_path, exist_ok=True)
        shutil.copyfile(config_path, os.path.join(self.train_encoder_output_path, "config.yaml"))

    def _load_transfer_learning_config(self, config_path):
        config = yaml.safe_load(open(config_path, 'r'))

        self.transfer_learning_train_datasets = config['train_datasets']
        self.transfer_learning_batch_size = config['batch_size']
        self.transfer_learning_num_workers = config['num_workers']
        self.transfer_learning_prefetch_factor = config['prefetch_factor']
        self.transfer_learning_pin_memory = bool(config['pin_memory'])
        self.transfer_learning_use_checkpoint = bool(config['use_checkpoint'])
        self.transfer_learning_use_val_subset = bool(config['use_val_subset'])
        self.transfer_learning_config_path = config_path

    def _load_linear_evaluation_config(self, config_path):
        config = yaml.safe_load(open(config_path, 'r'))

        self.linear_evaluation_train_datasets = config['train_datasets']
        self.linear_evaluation_batch_size = config['batch_size']
        self.linear_evaluation_num_workers = config['num_workers']
        self.linear_evaluation_prefetch_factor = config['prefetch_factor']
        self.linear_evaluation_pin_memory = bool(config['pin_memory'])
        self.linear_evaluation_use_checkpoint = bool(config['use_checkpoint'])
        self.linear_evaluation_use_val_subset = bool(config['use_val_subset'])
        self.linear_evaluation_config_path = config_path

    def _load_train_encoder_config(self, config_path):
        config = yaml.safe_load(open(config_path, 'r'))

        config['output_path'] += '/' if not config['output_path'].endswith('/') else ''
        config['datasets_path'] += '/' if not config['datasets_path'].endswith('/') else ''

        # Configs to class attributes
        if self.operation != "train_encoder":
            self.train_encoder_output_path = str(config['output_path'])
        self.train_encoder_datasets_path = str(config['datasets_path'])
        self.train_encoder_batch_size = int(config['batch_size'])
        self.train_encoder_num_epochs = int(config['num_epochs'])
        self.train_encoder_lr = float(config['lr'])
        self.train_encoder_weight_decay = float(config['weight_decay'])
        self.train_encoder_num_workers = int(config['num_workers'])
        self.train_encoder_model_name = str(config['model'])
        self.train_encoder_transform_resize = tuple(config['transform_resize'])
        self.train_encoder_train_datasets = list(config['train_datasets'])
        self.train_encoder_projection_head_mode = str(config['projection_head_mode'])
        self.train_encoder_temperature = float(config['temperature'])
        self.train_encoder_projection_dim = int(config['projection_dim'])
        self.train_encoder_warmup_epochs = int(config['warmup_epochs'])
        self.train_encoder_prefetch_factor = int(config['prefetch_factor'])
        self.train_encoder_pin_memory = bool(config['pin_memory'])
        self.train_encoder_use_checkpoint = bool(config['use_checkpoint'])
        self.train_encoder_pretrained = bool(config['pretrained'])
        self.train_encoder_use_val_subset = bool(config['use_val_subset'])

    def plot_fig(self, x, x_name, y, y_name, fig_name):
        plt.figure()
        plt.plot(x, y)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title(fig_name)

        output_path = None
        match self.operation:
            case "train_encoder": output_path = self.train_encoder_output_path
            case "linear_evaluation": output_path = self.linear_evaluation_output_path
            case "transfer_learning": output_path = self.transfer_learning_output_path

        os.makedirs(output_path + "figs", exist_ok=True)

        plt.savefig(os.path.join(output_path + "figs", f"{fig_name}.png"))
        plt.close()

    def write_on_log(self, text):
        time = strftime("%Y-%m-%d %H:%M:%S - ", localtime())

        output_path = None
        match self.operation:
            case "train_encoder": output_path = self.train_encoder_output_path
            case "linear_evaluation": output_path = self.linear_evaluation_output_path
            case "transfer_learning": output_path = self.transfer_learning_output_path

        mode = "w" if not os.path.exists(os.path.join(output_path, "log.txt")) else "a"

        with open(os.path.join(output_path, "log.txt"), mode) as file:
            file.write(time + text + "\n")

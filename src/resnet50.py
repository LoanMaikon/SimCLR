from torchvision.transforms import v2
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import shutil
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np
import os
from time import strftime, localtime
import json

class resnet50(nn.Module):
    def __init__(self, projection_head_mode):
        super(resnet50, self).__init__()
        self.projection_head_mode = projection_head_mode
        self.backbone = models.resnet50(weights=None) # Training from scratch
        self.projection_head = None

        self._load_projection_head()
    
    '''
    The paper defines the latent space of size 128
    '''
    def _load_projection_head(self):
        match self.projection_head_mode:
            case 'linear':
                self.projection_head = nn.Linear(self.backbone.fc.in_features, self.backbone.fc.in_features)

            case 'non-linear':
                self.projection_head = nn.Sequential(
                    nn.Linear(self.backbone.fc.in_features, self.backbone.fc.in_features, bias=False),
                    nn.BatchNorm1d(self.backbone.fc.in_features),
                    nn.ReLU(),
                    nn.Linear(self.backbone.fc.in_features, 128, bias=False),
                    nn.BatchNorm1d(128),
                )
            
            case 'none':
                self.projection_head = nn.Identity()
        
        if self.projection_head is None:
            raise ValueError("Projection head mode must be 'linear', 'non-linear', or 'none'.")
        
    def fit_projection_head(self):
        self.backbone.fc = self.projection_head
    
    def remove_projection_head(self):
        self.backbone.fc = nn.Identity()

    def fit_classifier(self, num_classes):
        pass

    def forward(self, x1, x2=None):
        x1 = self.backbone(x1)

        if x2 is not None:
            x2 = self.backbone(x2)
            return x1, x2

        return x1

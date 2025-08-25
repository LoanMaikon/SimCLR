import torch.utils.checkpoint
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

class resnet50(nn.Module):
    def __init__(self, projection_head_mode, projection_dim, use_checkpoint, pretrained):
        super(resnet50, self).__init__()
        self.projection_head_mode = projection_head_mode
        self.projection_dim = projection_dim
        self.use_checkpoint = use_checkpoint

        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)

        self.projection_head = None
        self.encoder_out_features = self.backbone.fc.in_features

        self._load_projection_head()

    '''
    Following https://github.com/google-research/simclr/blob/master/model_util.py
    '''
    def _load_projection_head(self):
        match self.projection_head_mode:
            case 'linear':
                self.projection_head = nn.Sequential(
                    nn.Linear(self.encoder_out_features, self.projection_dim, bias=False),
                    nn.BatchNorm1d(self.projection_dim),
                )


            case 'non-linear':
                self.projection_head = nn.Sequential(
                    nn.Linear(self.encoder_out_features, self.encoder_out_features, bias=False),
                    nn.BatchNorm1d(self.encoder_out_features),
                    nn.ReLU(),
                    nn.Linear(self.encoder_out_features, self.projection_dim, bias=False),
                    nn.BatchNorm1d(self.projection_dim),
                )
            
            case 'none':
                self.projection_head = nn.Identity()
        
        if self.projection_head is None:
            raise ValueError("Projection head mode must be 'linear', 'non-linear', or 'none'.")
    
    def freeze_encoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def fit_projection_head(self):
        self.backbone.fc = self.projection_head

        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def remove_projection_head(self):
        self.backbone.fc = nn.Identity()

    def fit_classifier_head(self, num_classes):
        self.backbone.fc = nn.Linear(self.encoder_out_features, num_classes, bias=True)

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x1, x2=None):
        def _forward_chunk(x):
            for name, module in self.backbone._modules.items():
                if name == 'fc':
                    break

                x = module(x)
                
            x = torch.flatten(x, 1)

            return x

        if x2 is None:
            if self.use_checkpoint:
                feats = torch.utils.checkpoint.checkpoint(_forward_chunk, x1, use_reentrant=False)
            else:
                feats = _forward_chunk(x1)

            return self.backbone.fc(feats)

        x = torch.cat([x1, x2], dim=0)

        if self.use_checkpoint:
            feats = torch.utils.checkpoint.checkpoint(_forward_chunk, x, use_reentrant=False)
        else:
            feats = _forward_chunk(x)

        proj = self.backbone.fc(feats)
        z1, z2 = proj.chunk(2, dim=0)

        return z1, z2

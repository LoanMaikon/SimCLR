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

class mobilenet_v3_small(nn.Module):
    def __init__(self, projection_head_mode, projection_dim, use_checkpoint, pretrained):
        super(mobilenet_v3_small, self).__init__()
        self.projection_head_mode = projection_head_mode
        self.projection_dim = projection_dim
        self.use_checkpoint = use_checkpoint

        if pretrained:
            self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.mobilenet_v3_small(weights=None)

        self.projection_head = None
        self.encoder_out_features = self.backbone.classifier[0].in_features

        self._load_projection_head()

    '''
    Following https://github.com/google-research/simclr/blob/master/model_util.py
    '''
    def _load_projection_head(self):
        match self.projection_head_mode:
            case 'linear':
                self.projection_head = nn.Sequential(
                    nn.Linear(self.encoder_out_features, self.projection_dim),
                    nn.BatchNorm1d(self.projection_dim),
                )


            case 'non-linear':
                self.projection_head = nn.Sequential(
                    nn.Linear(self.encoder_out_features, self.encoder_out_features),
                    nn.BatchNorm1d(self.encoder_out_features),
                    nn.ReLU(),
                    nn.Linear(self.encoder_out_features, self.projection_dim),
                    nn.BatchNorm1d(self.projection_dim),
                )
            
            case 'none':
                self.projection_head = nn.Identity()
        
        if self.projection_head is None:
            raise ValueError("Projection head mode must be 'linear', 'non-linear', or 'none'.")
    
    def freeze_encoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def fit_projection_head(self):
        self.backbone.classifier = self.projection_head

        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
    
    def remove_projection_head(self):
        self.backbone.classifier = nn.Identity()

    def fit_classifier_head(self, num_classes):
        self.backbone.classifier = nn.Linear(self.encoder_out_features, num_classes, bias=True)

        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def _forward_impl(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)

        return x
    
    def _forward_impl_checkpoint(self, x):
        for layer in self.backbone.features:
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)

        return x

    def forward(self, x1, x2=None):
        if x2 is None:
            if not self.use_checkpoint:
                return self._forward_impl(x1)
            else:
                return self._forward_impl_checkpoint(x1)

        x = torch.cat([x1, x2], dim=0)
        feats = None

        if not self.use_checkpoint:
            feats = self._forward_impl(x)
        else:
            feats = self._forward_impl_checkpoint(x)
        
        z1, z2 = feats.chunk(2, dim=0)

        return z1, z2

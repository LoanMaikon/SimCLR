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
import timm

class mobilenet_v4_small(nn.Module):
    def __init__(self, projection_head_mode, projection_dim, use_checkpoint, pretrained):
        super(mobilenet_v4_small, self).__init__()
        self.projection_head_mode = projection_head_mode
        self.projection_dim = projection_dim
        self.use_checkpoint = use_checkpoint

        if pretrained:
            self.backbone = timm.create_model("mobilenetv4_conv_small", pretrained=True)
        else:
            self.backbone = timm.create_model("mobilenetv4_conv_small", pretrained=False)

        self.projection_head = None
        self.encoder_out_features = self.backbone.classifier.in_features

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
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.blocks(x)
        x = self.backbone.global_pool(x)
        x = self.backbone.conv_head(x)
        x = self.backbone.norm_head(x)
        x = self.backbone.act2(x)
        x = self.backbone.flatten(x)
        x = self.backbone.classifier(x)

        return x
    
    def _forward_impl_checkpoint(self, x):
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)

        for block in self.backbone.blocks:
            for _block_layer in block.children():
                x = torch.utils.checkpoint.checkpoint(_block_layer, x, use_reentrant=False)

        x = self.backbone.global_pool(x)
        x = self.backbone.conv_head(x)
        x = self.backbone.norm_head(x)
        x = self.backbone.act2(x)
        x = self.backbone.flatten(x)
        x = self.backbone.classifier(x)

        return x

    def forward(self, x1, x2=None):
        if x2 is None:
            if not self.use_checkpoint:
                return self._forward_impl(x1)
            else:
                return self._forward_impl_checkpoint(x1)

        if not self.use_checkpoint:
            feats1 = self._forward_impl(x1)
            feats2 = self._forward_impl(x2)
        else:
            feats1 = self._forward_impl_checkpoint(x1)
            feats2 = self._forward_impl_checkpoint(x2)
        
        return feats1, feats2

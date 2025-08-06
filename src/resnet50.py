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
    def __init__(self, projection_head_mode, projection_dim):
        super(resnet50, self).__init__()
        self.projection_head_mode = projection_head_mode
        self.projection_dim = projection_dim

        # self.backbone = models.resnet50(weights=None) # Training from scratch
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

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

    def fit_projection_head(self):
        self.backbone.fc = self.projection_head
    
    def remove_projection_head(self):
        self.backbone.fc = nn.Identity()

    def fit_classifier_head(self, num_classes):
        self.backbone.fc = nn.Linear(self.encoder_out_features, num_classes, bias=True)

    def forward(self, x1, x2=None):
        def _forward_chunk(x):
            for name, module in self.backbone._modules.items():
                if name != 'fc':
                    x = module(x)
            x = torch.flatten(x, 1)

            return x
        
        x1_out = torch.utils.checkpoint.checkpoint(_forward_chunk, x1, use_reentrant=False)
        torch.cuda.empty_cache()
        
        if x2 is not None:
            x2_out = torch.utils.checkpoint.checkpoint(_forward_chunk, x2, use_reentrant=False)
            torch.cuda.empty_cache()
            
            return self.backbone.fc(x1_out), self.backbone.fc(x2_out)
        
        return self.backbone.fc(x1_out)

    # def forward(self, x1, x2=None):
    #     def process_input(x):
    #         def first_layers(x):
    #             x = self.backbone.conv1(x)
    #             x = self.backbone.bn1(x)
    #             x = self.backbone.relu(x)
    #             x = self.backbone.maxpool(x)

    #             return x
            
    #         def layer1(x):
    #             return self.backbone.layer1(x)
                
    #         def layer2(x):
    #             return self.backbone.layer2(x)
                
    #         def layer3(x):
    #             return self.backbone.layer3(x)
                
    #         def layer4(x):
    #             return self.backbone.layer4(x)
                
    #         def pool(x):
    #             x = self.backbone.avgpool(x)
    #             x = torch.flatten(x, 1)
    #             return x
                
    #         x = torch.utils.checkpoint.checkpoint(first_layers, x, use_reentrant=False, preserve_rng_state=False)
    #         torch.cuda.empty_cache()
    #         x = torch.utils.checkpoint.checkpoint(layer1, x, use_reentrant=False, preserve_rng_state=False)
    #         torch.cuda.empty_cache()
    #         x = torch.utils.checkpoint.checkpoint(layer2, x, use_reentrant=False, preserve_rng_state=False)
    #         torch.cuda.empty_cache()
    #         x = torch.utils.checkpoint.checkpoint(layer3, x, use_reentrant=False, preserve_rng_state=False)
    #         torch.cuda.empty_cache()
    #         x = torch.utils.checkpoint.checkpoint(layer4, x, use_reentrant=False, preserve_rng_state=False)
    #         torch.cuda.empty_cache()
    #         x = torch.utils.checkpoint.checkpoint(pool, x, use_reentrant=False, preserve_rng_state=False)
    #         torch.cuda.empty_cache()
            
    #         return x

    #     x1 = self.backbone.fc(process_input(x1))

    #     torch.cuda.empty_cache()

    #     if x2 is not None:
    #         x2 = self.backbone.fc(process_input(x2))

    #         torch.cuda.empty_cache()

    #         return x1, x2
        
    #     return x1

    # def forward(self, x1, x2=None):
    #     modules = list(self.backbone.children())[:-1]
    #     chunks = 16

    #     x1_feat = torch.utils.checkpoint.checkpoint_sequential(modules, chunks, x1).flatten(1)
    #     out1 = self.backbone.fc(x1_feat)
    #     torch.cuda.empty_cache()

    #     if x2 is not None:
    #         x2_feat = torch.utils.checkpoint.checkpoint_sequential(modules, chunks, x2).flatten(1)
    #         out2 = self.backbone.fc(x2_feat)
    #         torch.cuda.empty_cache()

    #         return out1, out2
    #     return out1

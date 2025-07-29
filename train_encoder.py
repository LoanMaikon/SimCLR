from torchvision.transforms import v2
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np
import os

from src.Model import Model

def main():
    args = get_args()
    
    model = Model(config_path=args.config, gpu_index=args.gpu, operation="train_encoder")



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="Path to the config file", required=True)
    parser.add_argument("--gpu", type=int, help="GPU index to use", required=True)

    return parser.parse_args()

if __name__ == "__main__":
    main()

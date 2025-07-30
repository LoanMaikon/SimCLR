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

    train(model)

'''
Selecting the epoch with the lowest validation loss
'''
def train(model):
    model.write_on_log("Starting training...")

    best_val_loss = float('inf')
    for epoch in range(model.get_train_encoder_num_epochs()):
        model.write_on_log(f"Epoch {epoch + 1}/{model.get_train_encoder_num_epochs()}")

        model.model_to_train()
        
        epoch_train_loss = 0.0
        for batch in model.get_train_dataloader():
            z1, z2 = model.model_infer(batch[0], batch[1])

            loss = model.get_criterion(z1, z2)
            loss.backward()
            epoch_train_loss += loss.item()

            model.get_optimizer().step()
            model.get_optimizer().zero_grad()

        epoch_train_loss /= len(model.get_train_dataloader())
        model.write_on_log(f"Training loss: {epoch_train_loss:.4f}")

        model.model_to_eval()

        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in model.get_val_dataloader():
                z1, z2 = model.model_infer(batch[0], batch[1])

                loss = model.get_criterion(z1, z2)
                epoch_val_loss += loss.item()
        epoch_val_loss /= len(model.get_val_dataloader())
        model.write_on_log(f"Validation loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            model.write_on_log(f"Validation loss improved from {best_val_loss:.4f} to {epoch_val_loss:.4f}. Saving model...")
            best_val_loss = epoch_val_loss
            model.save_model()

        model.write_on_log(f"")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="Path to the config file", required=True)
    parser.add_argument("--gpu", type=int, help="GPU index to use", required=True)

    return parser.parse_args()

if __name__ == "__main__":
    main()

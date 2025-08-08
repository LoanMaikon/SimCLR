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
Selecting the epoch with the lowest training loss
'''
def train(model):
    model.write_on_log(f"Starting training...")

    scaler = torch.amp.GradScaler('cuda' if model.get_gpu_index() is not None else 'cpu')

    best_train_loss = float('inf')
    best_val_loss = float('inf')
    for epoch in range(model.get_train_encoder_num_epochs()):
        model.write_on_log(f"Epoch {epoch + 1}/{model.get_train_encoder_num_epochs()}")

        model.model_to_train()
        
        epoch_train_loss = 0.0

        for batch in model.get_train_dataloader():
            model.get_optimizer().zero_grad()

            with torch.amp.autocast('cuda' if model.get_gpu_index() is not None else 'cpu'):
                z1, z2 = model.model_infer(batch[0], batch[1])

                loss = model.apply_criterion(z1, z2)

            scaler.scale(loss).backward()

            scaler.step(model.get_optimizer())

            scaler.update()

            epoch_train_loss += loss.item()

            torch.cuda.empty_cache()

        epoch_train_loss /= len(model.get_train_dataloader())
        model.write_on_log(f"Training loss: {epoch_train_loss:.4f}")

        if model.has_validation_set(): # Datasets other than ImageNet
            model.model_to_eval()

            val_loss = 0.0
            with torch.no_grad():
                for batch in model.get_validation_dataloader():
                    z1, z2 = model.model_infer(batch[0], batch[1])
                    loss = model.apply_criterion(z1, z2)
                    val_loss += loss.item()

            val_loss /= len(model.get_validation_dataloader())
            model.write_on_log(f"Validation loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                model.write_on_log(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
                best_val_loss = val_loss
                model.save_model()

        else: # We use ImageNet validation set as a test set following proposed protocol
            if epoch_train_loss < best_train_loss:
                model.write_on_log(f"Training loss improved from {best_train_loss:.4f} to {epoch_train_loss:.4f}. Saving model...")
                best_train_loss = epoch_train_loss
                model.save_model()
        
        model.get_scheduler().step()

        model.write_on_log(f"")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="Path to the config file", required=True)
    parser.add_argument("--gpu", type=int, help="GPU index to use", required=False)

    return parser.parse_args()

if __name__ == "__main__":
    main()

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

    train_losses = []
    val_losses = []
    val_accs = []
    lrs = []

    scaler = torch.amp.GradScaler()

    best_train_loss = float('inf')
    best_val_loss = float('inf')
    for epoch in range(1, model.get_train_encoder_num_epochs() + 1):
        lrs.append(model.get_learning_rate())
        model.write_on_log(f"Epoch {epoch}/{model.get_train_encoder_num_epochs()}")

        model.model_to_train()
        
        epoch_train_loss = 0.0
        epoch_train_samples = 0

        current_batch = 0

        accumulated_x1 = []
        accumulated_x2 = []

        len_train_dataloader = len(model.get_train_dataloader())
        for idx, batch in enumerate(model.get_train_dataloader()):
            current_batch += 1

            accumulated_x1.append(batch[0])
            accumulated_x2.append(batch[1])

            if (current_batch >= model.train_encoder_batch_size // model.train_encoder_worker_batch_size) or (idx >= len_train_dataloader - 1):
                x1 = torch.cat(accumulated_x1, dim=0)
                x2 = torch.cat(accumulated_x2, dim=0)

                model.get_optimizer().zero_grad()

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    z1, z2 = model.model_infer(x1, x2)
                    loss = model.apply_criterion(z1, z2)
                
                epoch_train_loss += loss.item() * x1.size(0)
                epoch_train_samples += x1.size(0)

                scaler.scale(loss).backward()
                scaler.step(model.get_optimizer())

                scaler.update()

                current_batch = 0
                accumulated_x1 = []
                accumulated_x2 = []
            else:
                continue

        epoch_train_loss /= epoch_train_samples
        train_losses.append(epoch_train_loss)
        model.write_on_log(f"Training loss: {epoch_train_loss:.4f}")

        if epoch_train_loss < best_train_loss:
            model.write_on_log(f"Training loss improved from {best_train_loss:.4f} to {epoch_train_loss:.4f}. Saving model...")
            best_train_loss = epoch_train_loss
            model.save_model()
        
        if epoch % model.train_encoder_save_every == 0 or epoch == model.get_train_encoder_num_epochs():
            model.save_model(f"model_{epoch}.pth")
        
        model.get_scheduler().step()

        model.write_on_log(f"")

        model.plot_fig(
            x=range(1, len(train_losses) + 1),
            x_name="Epochs",
            y=train_losses,
            y_name="Training Loss",
            fig_name="train_loss"
        )

        if model.get_use_val_subset():
            model.plot_fig(
                x=range(1, len(val_losses) + 1),
                x_name="Epochs",
                y=val_losses,
                y_name="Validation Loss",
                fig_name="val_loss"
            )

            model.plot_fig(
                x=range(1, len(val_accs) + 1),
                x_name="Epochs",
                y=val_accs,
                y_name="Validation Accuracy",
                fig_name="val_acc"
            )

        model.plot_fig(
            x=range(1, len(lrs) + 1),
            x_name="Epochs",
            y=lrs,
            y_name="Learning Rate",
            fig_name="lr"
        )

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="Path to the config file", required=True)
    parser.add_argument("--gpu", type=int, help="GPU index to use", required=True)

    return parser.parse_args()

if __name__ == "__main__":
    main()

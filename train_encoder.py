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
    for epoch in range(model.get_train_encoder_num_epochs()):
        lrs.append(model.get_learning_rate())
        model.write_on_log(f"Epoch {epoch + 1}/{model.get_train_encoder_num_epochs()}")

        model.model_to_train()
        
        epoch_train_loss = 0.0
        epoch_train_samples = 0

        for batch in model.get_train_dataloader():
            model.get_optimizer().zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.float16):
                z1, z2 = model.model_infer(batch[0], batch[1])
                loss = model.apply_criterion(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(model.get_optimizer())

            scaler.update()

            epoch_train_loss += loss.item() * batch[0].size(0)
            epoch_train_samples += batch[0].size(0)

        epoch_train_loss /= epoch_train_samples
        train_losses.append(epoch_train_loss)
        model.write_on_log(f"Training loss: {epoch_train_loss:.4f}")

        if model.get_use_val_subset():
            model.model_to_eval()

            val_loss = 0.0
            total_val_samples = 0
            total_val_preds = 0
            correct_val_samples = 0

            with torch.no_grad():
                for batch in model.get_val_dataloader():
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        z1, z2 = model.model_infer(batch[0], batch[1])
                        loss = model.apply_criterion(z1, z2)

                    predicted, targets = _get_prediction_and_target(z1, z2, model.get_device())

                    val_loss += loss.item() * batch[0].size(0)
                    total_val_samples += batch[0].size(0)

                    correct_val_samples += (predicted == targets).sum().item()
                    total_val_preds += predicted.size(0)

            val_loss /= total_val_samples
            val_acc = correct_val_samples / total_val_preds

            val_losses.append(val_loss)
            val_accs.append(val_acc)

            model.write_on_log(f"Validation loss: {val_loss:.4f}")
            model.write_on_log(f"Validation accuracy: {val_acc:.4f}")

            if val_loss < best_val_loss:
                model.write_on_log(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
                best_val_loss = val_loss
                model.save_model()

        else:
            if epoch_train_loss < best_train_loss:
                model.write_on_log(f"Training loss improved from {best_train_loss:.4f} to {epoch_train_loss:.4f}. Saving model...")
                best_train_loss = epoch_train_loss
                model.save_model()
        
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

def _get_prediction_and_target(z1, z2, device=None):
    z1_cpu = z1.detach().cpu()
    z2_cpu = z2.detach().cpu()

    z1_cpu = torch.nn.functional.normalize(z1_cpu, dim=1)
    z2_cpu = torch.nn.functional.normalize(z2_cpu, dim=1)

    batch_size = z1_cpu.size(0)

    sim = torch.matmul(z1_cpu, z2_cpu.T)

    # For each z1, find the most similar z2 and vice-versa
    pred_1_to_2 = sim.argmax(dim=1)
    pred_2_to_1 = sim.T.argmax(dim=1)

    targets = torch.arange(batch_size, dtype=torch.long)

    predicted = torch.cat([pred_1_to_2, pred_2_to_1], dim=0)
    targets = torch.cat([targets, targets], dim=0)

    return predicted, targets


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="Path to the config file", required=True)
    parser.add_argument("--gpu", type=int, help="GPU index to use", required=True)

    return parser.parse_args()

if __name__ == "__main__":
    main()

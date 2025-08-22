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
    executions_names = get_executions_names(args.train_dir)

    for execution_name in executions_names:
        encoder_config = _get_encoder_config_path_by_execution_name(args.train_dir, execution_name)

        for label_fraction, num_epochs, lr, weight_decay in zip(args.label_fractions, args.num_epochs, args.lrs, args.weight_decays):
            model = Model(config_path=args.config,
                          gpu_index=args.gpu,
                          operation="linear_evaluation",
                          execution_name=execution_name, 
                          label_fraction=label_fraction,
                          lr=lr,
                          weight_decay=weight_decay,
                          num_epochs=num_epochs,
                          encoder_config=encoder_config)

            model.write_on_log(f"Label fraction: {label_fraction} Num epochs: {num_epochs} Learning rate: {lr} Weight decay: {weight_decay}\n")

            train(model, label_fraction, num_epochs, lr, weight_decay)
            test(model, label_fraction, num_epochs, lr, weight_decay)

'''
Selecting the epoch with the lowest validation loss for datasets where it is available.
For datasets without validation set, the epoch with the lowest training loss is selected.
'''
def train(model, label_fraction, num_epochs, lr, weight_decay):
    model.write_on_log(f"Starting training...")

    scaler = torch.amp.GradScaler()

    best_val_loss = float('inf')
    best_train_loss = float('inf')

    train_losses = []
    val_losses = []
    lrs = []

    for epoch in range(model.get_linear_evaluation_num_epochs()):
        lrs.append(model.get_learning_rate())
        model.write_on_log(f"Epoch {epoch + 1}/{model.get_linear_evaluation_num_epochs()}")

        model.model_to_train()
        
        epoch_train_loss = 0.0
        for batch in model.get_train_dataloader():
            model.get_optimizer().zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.float16) if model.get_device().type == 'cuda' else torch.autocast('cpu'):
                z1 = model.model_infer(batch[0])
                targets = batch[1].to(model.get_device())
                loss = model.apply_criterion(z1, targets)

            scaler.scale(loss).backward()
            scaler.step(model.get_optimizer())
            scaler.update()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(model.get_train_dataloader())
        train_losses.append(epoch_train_loss)
        model.write_on_log(f"Training loss: {epoch_train_loss:.4f}")

        if model.has_validation_set():
            model.model_to_eval()

            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch in model.get_val_dataloader():
                    with torch.amp.autocast('cuda', dtype=torch.float16) if model.get_device().type == 'cuda' else torch.autocast('cpu'):
                        z1 = model.model_infer(batch[0])
                        targets = batch[1].to(model.get_device())
                        loss = model.apply_criterion(z1, targets)

                    epoch_val_loss += loss.item()
            
            epoch_val_loss /= len(model.get_val_dataloader())
            val_losses.append(epoch_val_loss)
            model.write_on_log(f"Validation loss: {epoch_val_loss:.4f}")

            if epoch_val_loss < best_val_loss:
                model.write_on_log(f"Validation loss improved from {best_val_loss:.4f} to {epoch_val_loss:.4f}. Saving model...")
                best_val_loss = epoch_val_loss
                model.save_model()

        else:
            if epoch_train_loss < best_train_loss:
                model.write_on_log(f"Training loss improved from {best_train_loss:.4f} to {epoch_train_loss:.4f}. Saving model...")
                best_train_loss = epoch_train_loss
                model.save_model()
        
        model.write_on_log(f"")

        model.plot_fig(
            x=range(1, len(train_losses) + 1),
            x_name="Epochs",
            y=train_losses,
            y_name="Training Loss",
            fig_name=f"train_loss_lf_{label_fraction}_ne_{num_epochs}_lr_{lr}_wd_{weight_decay}.png"
        )

        if model.has_validation_set():
            model.plot_fig(
                x=range(1, len(val_losses) + 1),
                x_name="Epochs",
                y=val_losses,
                y_name="Validation Loss",
                fig_name=f"val_loss_lf_{label_fraction}_ne_{num_epochs}_lr_{lr}_wd_{weight_decay}.png"
            )

def test(model, label_fraction, num_epochs, lr, weight_decay):
    model.write_on_log(f"Starting testing...")

    model.model_to_eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in model.get_test_dataloader():
            with torch.amp.autocast('cuda', dtype=torch.float16) if model.get_device().type == 'cuda' else torch.autocast('cpu'):
                z1 = model.model_infer(batch[0])
                targets = batch[1].to(model.get_device())

            output = nn.functional.softmax(z1, dim=1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(output.cpu().numpy())

    model.save_results(
        targets=all_targets,
        all_predictions=all_predictions,
        json_name=f"results_lb_{label_fraction}_ne_{num_epochs}_lr_{lr}_wd_{weight_decay}.json"
    )

    model.write_on_log(f"Testing completed\n")

def get_executions_names(train_dir):
    return sorted(os.listdir(train_dir))

def _get_encoder_config_path_by_execution_name(train_dir, execution_name):
    return f"{train_dir}/{execution_name}/train_encoder/config.yaml"

def get_args():
    parser = argparse.ArgumentParser(description="Linear Evaluation Training")
    parser.add_argument("--train_dir", type=str, help="Path to encoder training directory", required=True)
    parser.add_argument("--config", type=str, help="Path to config file", required=True)
    parser.add_argument("--gpu", type=int, help="GPU index", required=True)

    return parser.parse_args()

if __name__ == "__main__":
    main()

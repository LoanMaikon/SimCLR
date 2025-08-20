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

LABEL_FRACTIONS = None
NUM_EPOCHS = None
LRS = None
WEIGHT_DECAYS = None

def main():
    args = get_args()
    executions_names = get_executions_names(args.config)
    _set_configs(args.config)

    for execution_name in executions_names:
        for label_fraction, num_epochs, lr, weight_decay in zip(LABEL_FRACTIONS, NUM_EPOCHS, LRS, WEIGHT_DECAYS):
            model = Model(config_path=args.config, gpu_index=args.gpu, operation="transfer_learning", execution_name=execution_name, label_fraction=label_fraction, 
                          num_epochs=num_epochs, lr=lr, weight_decay=weight_decay)

            model.write_on_log(f"Label fraction: {label_fraction} Num epochs: {num_epochs} Learning rate: {lr} Weight decay: {weight_decay}\n")

            train(model)
            test(model)

'''
Selecting the epoch with the lowest validation loss for datasets where it is available.
For datasets without validation set, the epoch with the lowest training loss is selected.
'''

def train(model):
    model.write_on_log("Starting training...")

    scaler = torch.amp.GradScaler()

    accumulation_steps = max(1, model.get_transfer_learning_batch_size() // model.get_chunk_size())

    optimizer = model.get_optimizer()
    best_val_loss = float('inf')
    best_train_loss = float('inf')

    train_losses = []
    val_losses = []
    lrs = []

    for epoch in range(model.get_transfer_learning_num_epochs()):
        lrs.append(model.get_learning_rate())
        model.write_on_log(f"Epoch {epoch + 1}/{model.get_transfer_learning_num_epochs()}")

        model.model_to_train()

        epoch_train_loss_sum = 0.0
        total_train_samples = 0
        chunk_counter = 0

        optimizer.zero_grad()
        for batch in model.get_train_dataloader():
            inputs = batch[0].to(model.get_device())
            targets = batch[1].to(model.get_device())

            for i in range(0, len(inputs), model.get_chunk_size()):
                chunk_inputs = inputs[i:i + model.get_chunk_size()]
                chunk_targets = targets[i:i + model.get_chunk_size()]

                if len(chunk_inputs) == 0:
                    continue

                chunk_counter += 1

                with torch.amp.autocast('cuda') if model.get_device().type == 'cuda' else torch.autocast('cpu'):
                    preds = model.model_infer(chunk_inputs)
                    raw_loss = model.apply_criterion(preds, chunk_targets)
                scaled = raw_loss / accumulation_steps
                scaler.scale(scaled).backward()

                chunk_size = chunk_inputs.size(0)
                epoch_train_loss_sum += raw_loss.item() * chunk_size
                total_train_samples += chunk_size

                if chunk_counter % accumulation_steps == 0 or i + model.get_chunk_size() >= len(inputs):
                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad()

        epoch_train_loss = (epoch_train_loss_sum / total_train_samples)
        train_losses.append(epoch_train_loss)
        model.write_on_log(f"Training loss: {epoch_train_loss:.4f}")

        if model.has_validation_set():
            model.model_to_eval()
            epoch_val_loss_sum = 0.0
            total_val_samples = 0

            with torch.no_grad():
                for batch in model.get_val_dataloader():
                    inputs = batch[0].to(model.get_device())
                    targets = batch[1].to(model.get_device())

                    with torch.amp.autocast('cuda') if model.get_device().type == 'cuda' else torch.autocast('cpu'):
                        preds = model.model_infer(inputs)
                        raw_loss = model.apply_criterion(preds, targets)

                    batch_size = inputs.size(0)
                    epoch_val_loss_sum += raw_loss.item() * batch_size
                    total_val_samples += batch_size

            epoch_val_loss = (epoch_val_loss_sum / total_val_samples) if total_val_samples > 0 else 0.0
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

        model.write_on_log("")
    
    model.plot_fig(
        x=range(1, model.get_train_encoder_num_epochs() + 1),
        x_name="Epochs",
        y=train_losses,
        y_name="Training Loss",
        fig_name="train_loss.png"
    )

    if model.has_validation_set():
        model.plot_fig(
            x=range(1, model.get_train_encoder_num_epochs() + 1),
            x_name="Epochs",
            y=val_losses,
            y_name="Validation Loss",
            fig_name="val_loss.png"
        )

def test(model):
    model.write_on_log(f"Starting testing...")

    model.model_to_eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in model.get_test_dataloader():
            with torch.amp.autocast('cuda') if model.get_device().type == 'cuda' else torch.autocast('cpu'):
                z1 = model.model_infer(batch[0])
                targets = batch[1].to(model.get_device())

            output = nn.functional.softmax(z1, dim=1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(output.cpu().numpy())

    model.save_results(
        targets=all_targets,
        all_predictions=all_predictions,
    )

    model.write_on_log(f"Testing completed\n")

def _set_configs(config):
    global LABEL_FRACTIONS, NUM_EPOCHS, LRS, WEIGHT_DECAYS
    transfer_learning_config = yaml.safe_load(open(config, 'r'))
    LABEL_FRACTIONS = transfer_learning_config['label_fractions']
    NUM_EPOCHS = transfer_learning_config['num_epochs']
    LRS = transfer_learning_config['lr']
    WEIGHT_DECAYS = transfer_learning_config['weight_decay']

def get_executions_names(config):
    transfer_learning_config = yaml.safe_load(open(config, 'r'))
    train_encoder_config = transfer_learning_config['encoder_config']
    train_encoder_config = yaml.safe_load(open(train_encoder_config, 'r'))
    train_encoder_output_path = train_encoder_config['output_path']

    return sorted(os.listdir(train_encoder_output_path))

def get_args():
    parser = argparse.ArgumentParser(description="Linear Evaluation Training")
    parser.add_argument("--config", type=str, help="Path to config file", required=True)
    parser.add_argument("--gpu", type=int, help="GPU index", required=False)

    return parser.parse_args()

if __name__ == "__main__":
    main()

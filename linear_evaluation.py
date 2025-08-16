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

def main():
    args = get_args()
    executions_names = get_executions_names(args.config)
    _set_label_fractions_and_num_epochs(args.config)

    for execution_name in executions_names:
        for label_fraction, num_epochs in zip(LABEL_FRACTIONS, NUM_EPOCHS):
            model = Model(config_path=args.config, gpu_index=args.gpu, operation="linear_evaluation", execution_name=execution_name, label_fraction=label_fraction)
            model.set_num_epochs(num_epochs)

            model.write_on_log(f"Label fraction: {label_fraction}\nNum epochs: {num_epochs}\n")

            train(model)
            test(model)

'''
Selecting the epoch with the lowest validation loss for datasets where it is available.
For datasets without validation set, the epoch with the lowest training loss is selected.
'''
def train(model):
    model.write_on_log(f"Starting training...")

    scaler = torch.amp.GradScaler('cuda' if model.get_gpu_index() is not None else 'cpu')

    best_val_loss = float('inf')
    best_train_loss = float('inf')
    for epoch in range(model.get_linear_evaluation_num_epochs()):
        model.write_on_log(f"Epoch {epoch + 1}/{model.get_linear_evaluation_num_epochs()}")

        model.model_to_train()
        
        epoch_train_loss = 0.0
        for batch in model.get_train_dataloader():
            model.get_optimizer().zero_grad()

            with torch.amp.autocast('cuda' if model.get_gpu_index() is not None else 'cpu'):
                z1 = model.model_infer(batch[0])
                targets = batch[1].to(model.get_device())

                loss = model.apply_criterion(z1, targets)

            scaler.scale(loss).backward()
            scaler.step(model.get_optimizer())
            scaler.update()

            epoch_train_loss += loss.item()

            torch.cuda.empty_cache()

        epoch_train_loss /= len(model.get_train_dataloader())
        model.write_on_log(f"Training loss: {epoch_train_loss:.4f}")

        if model.has_validation_set(): # Datasets other than ImageNet
            model.model_to_eval()

            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch in model.get_val_dataloader():
                    with torch.amp.autocast('cuda' if model.get_gpu_index() is not None else 'cpu'):
                        z1 = model.model_infer(batch[0])
                        targets = batch[1].to(model.get_device())

                        loss = model.apply_criterion(z1, targets)
                    epoch_val_loss += loss.item()
            
            epoch_val_loss /= len(model.get_val_dataloader())
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

def test(model):
    model.write_on_log(f"Starting testing...")

    model.model_to_eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in model.get_test_dataloader():
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

def _set_label_fractions_and_num_epochs(config):
    linear_evaluation_config = yaml.safe_load(open(config, 'r'))

    global LABEL_FRACTIONS, NUM_EPOCHS
    LABEL_FRACTIONS = linear_evaluation_config['label_fractions']
    NUM_EPOCHS = linear_evaluation_config['num_epochs']

    assert len(LABEL_FRACTIONS) == len(NUM_EPOCHS), "The number of label fractions must match the number of epochs."

def get_executions_names(config):
    linear_evaluation_config = yaml.safe_load(open(config, 'r'))
    train_encoder_config = linear_evaluation_config['encoder_config']
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

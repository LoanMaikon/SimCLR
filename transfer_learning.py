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
            model = Model(config_path=args.config, gpu_index=args.gpu, operation="transfer_learning", execution_name=execution_name, label_fraction=label_fraction)
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

    use_cuda = model.get_gpu_index() is not None
    scaler = None
    if use_cuda:
        scaler = torch.amp.GradScaler('cuda')

    accumulation_steps = max(1, model.get_transfer_learning_batch_size() // model.get_chunk_size())

    best_val_loss = float('inf')
    best_train_loss = float('inf')
    for epoch in range(model.get_transfer_learning_num_epochs()):
        model.write_on_log(f"Epoch {epoch + 1}/{model.get_transfer_learning_num_epochs()}")

        model.model_to_train()
        
        epoch_train_loss = 0.0
        batch_count = 0
        optimizer = model.get_optimizer()
        optimizer.zero_grad()
        for batch in model.get_train_dataloader():
            inputs = batch[0].to(model.get_device())
            targets = batch[1].to(model.get_device())

            for i in range(0, len(inputs), model.get_chunk_size()):
                chunk_inputs = inputs[i:i + model.get_chunk_size()]
                chunk_targets = targets[i:i + model.get_chunk_size()]

                if len(chunk_inputs) == 0:
                    continue

                batch_count += 1

                if use_cuda:
                    with torch.amp.autocast("cuda"):
                        z1 = model.model_infer(chunk_inputs)
                        loss = model.apply_criterion(z1, chunk_targets) / accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    z1 = model.model_infer(chunk_inputs)
                    loss = model.apply_criterion(z1, chunk_targets) / accumulation_steps
                    loss.backward()

                epoch_train_loss += loss.item() 

                if batch_count % accumulation_steps == 0:
                    if use_cuda:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

        if batch_count % accumulation_steps != 0:
            if use_cuda:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        epoch_train_loss = epoch_train_loss / batch_count if batch_count > 0 else 0.0
        model.write_on_log(f"Training loss: {epoch_train_loss:.4f}")

        if model.has_validation_set():
            model.model_to_eval()

            epoch_val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in model.get_val_dataloader():
                    inputs = batch[0].to(model.get_device())
                    targets = batch[1].to(model.get_device())
                    if use_cuda:
                        with torch.amp.autocast("cuda"):
                            z1 = model.model_infer(inputs)
                            loss = model.apply_criterion(z1, targets) / accumulation_steps
                    else:
                        z1 = model.model_infer(inputs)
                        loss = model.apply_criterion(z1, targets) / accumulation_steps
                    epoch_val_loss += loss.item()
                    val_count += 1
            
            epoch_val_loss /= val_count if val_count > 0 else 1.0
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

    use_cuda = model.get_gpu_index() is not None

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in model.get_test_dataloader():
            if use_cuda:
                with torch.amp.autocast('cuda'):
                    z1 = model.model_infer(batch[0])
                    targets = batch[1].to(model.get_device())

                    output = nn.functional.softmax(z1, dim=1)
            else:
                z1 = model.model_infer(batch[0])
                targets = batch[1]

                output = nn.functional.softmax(z1, dim=1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(output.cpu().numpy())

    model.save_results(
        targets=all_targets,
        all_predictions=all_predictions,
    )

    model.write_on_log(f"Testing completed\n")

def _set_label_fractions_and_num_epochs(config):
    global LABEL_FRACTIONS, NUM_EPOCHS
    transfer_learning_config = yaml.safe_load(open(config, 'r'))
    LABEL_FRACTIONS = transfer_learning_config['label_fractions']
    NUM_EPOCHS = transfer_learning_config['num_epochs']

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

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
    executions_names = get_executions_names(args.config)

    for execution_name in executions_names:
        model = Model(config_path=args.config, gpu_index=args.gpu, operation="linear_evaluation", execution_name=execution_name)

        train(model)
        test(model)
    
def train(model):
    model.write_on_log(f"Starting training...")

    best_val_loss = float('inf')
    for epoch in range(model.get_linear_evaluation_num_epochs()):
        model.write_on_log(f"Epoch {epoch + 1}/{model.get_linear_evaluation_num_epochs()}")

        model.model_to_train()
        
        epoch_train_loss = 0.0
        for batch in model.get_train_dataloader():
            model.get_optimizer().zero_grad()

            z1 = model.model_infer(batch[0])
            targets = batch[1].to(model.get_device())

            loss = model.apply_criterion(z1, targets)
            loss.backward()
            epoch_train_loss += loss.item()

            model.get_optimizer().step()

        epoch_train_loss /= len(model.get_train_dataloader())
        model.write_on_log(f"Training loss: {epoch_train_loss:.4f}")

        model.model_to_eval()

        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in model.get_val_dataloader():
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

        model.write_on_log(f"")

def test(model):
    model.write_on_log(f"Starting testing...")

    model.model_to_eval()

    all_targets = []
    top_5_predictions = []
    top_1_predictions = []

    with torch.no_grad():
        for batch in model.get_test_dataloader():
            z1 = model.model_infer(batch[0])
            targets = batch[1].to(model.get_device())

            output = nn.functional.softmax(z1, dim=1)

            _, top_5 = output.topk(5, dim=1)
            top_5 = top_5.cpu().numpy()
            top_5_predictions.append(top_5)

            top_1 = output.argmax(dim=1, keepdim=True).cpu().numpy()
            top_1_predictions.append(top_1)

            all_targets.append(targets.cpu().numpy())

    model.save_results(
        targets=all_targets,
        top_5_predictions=top_5_predictions,
        top_1_predictions=top_1_predictions,
    )

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

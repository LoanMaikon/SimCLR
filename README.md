<div align="center">

# SimCLR

SimCLR implementation with scripts for: self-supervised encoder pretraining, linear evaluation, fine-tuning and transfer learning across multiple datasets.

</div>

## 1. Overview
This repository contains:
- `train_encoder.py`: Pretraining of the encoder.
- `linear_evaluation.py`: Freezes the encoder and trains a linear classifier.
- `transfer_learning.py`: If done in the same dataset as the encoder, a fine-tuning is performed. If not, a transfer learning is performed.
- `configs/*.yaml`: Configs.
- `src/`: Source code.
- `tools/create_datasets.py`: Script to download datasets.

## 2. Steps

In `configs/train_encoder_imagenet` there is an example of pretraining of the encoder where

<pre>
<output_path> /disk0/lmk22/simclr_output/train_encoder_imagenet
<datasets_path> /mnt/raid1/lmk22/datasets
<train_datasets> [imagenet]
<batch_size> 512 # Paper uses 4096
<num_epochs> 100 # The paper uses 100 epochs
<lr> 1.697
<weight_decay> 0.000001
<num_workers> 40
<prefetch_factor> 2
<model> resnet50
<transform_resize> [224, 224]
<projection_head_mode> non-linear # linear, non-linear or none
<temperature> 0.5 # The paper tells us that the optimal and stable temperature is 0.5
<projection_dim> 128 # The paper defines the latent space of size 128
<warmup_epochs> 10 # The paper uses 10 epochs to warm up the learning rate
<pin_memory> False # Set to True if you have enough memory. Larger datasets will not benefit from this
<use_checkpoint> True # Recommended for memory efficiency
<pretrained> False # Set to True if you want to use a pretrained model
<use_val_subset> False # Set to True if you want to test hyperparameters on a validation subset
<optimizer> lars # The paper uses LARS optimizer
</pre>
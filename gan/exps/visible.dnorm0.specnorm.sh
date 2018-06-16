#!/bin/bash

# specnorm paper use beta=(0.0, 0.999)
source activate pytorch-env-py36
env
cd ../..
NAME=visible_dnorm0_specnorm
python task_launcher.py \
--name=${NAME} \
--batch_size=64 \
--epochs=600 \
--z_dim=62 \
--beta1=0.0 \
--iterator=iterators/cifar10.py \
--network=networks/model_resnet_specnorm.py \
--save_every=20 \
--mixup=pixel \
--alpha=0.2 \
--update_g_every=5 \
--dnorm=0.0 \
--resume=auto

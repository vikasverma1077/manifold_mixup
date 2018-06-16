#!/bin/bash

env
source activate pytorch-env-py36
cd ../..
NAME=hidden_dnorm0.5_pp
python task_launcher.py \
--name=${NAME} \
--batch_size=64 \
--epochs=500 \
--z_dim=62 \
--beta1=0.0 \
--iterator=iterators/cifar10.py \
--network=networks/model_resnet_preproc.py \
--save_every=10 \
--dnorm=0.5 \
--mixup=vh1 \
--alpha=0.2 \
--update_g_every=1 \
--resume=auto

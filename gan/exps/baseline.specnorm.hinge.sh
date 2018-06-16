#!/bin/bash

# specnorm paper use beta=(0.0, 0.999)
source activate pytorch-env-py36
env
cd ../..
BS=64
NAME=baseline_specnorm_hinge
python task_launcher.py \
--name=${NAME} \
--batch_size=${BS} \
--epochs=500 \
--z_dim=62 \
--beta1=0.0 \
--iterator=iterators/cifar10.py \
--network=networks/model_resnet_specnorm_nosigm.py \
--save_every=10 \
--dnorm=0.0 \
--update_g_every=5 \
--loss=hinge \
--resume=auto

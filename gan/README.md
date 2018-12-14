# GAN mixup

**NOTE: This code base has gone through some major changes since its initial submission to NIPS. It will be updated soon, but for now, this is the old code.**

Official PyTorch implementation of 
[Manifold Mixup: Encouraging Meaningful On-Manifold Interpolation as a Regularizer](https://arxiv.org/abs/1611.04076).

Here we use mixup to help regularise the discriminator. We propose the following objective function for the discriminator:

<img src="https://user-images.githubusercontent.com/2417792/41466879-22111e32-7072-11e8-83d4-644b46d4685a.png" width=600 />

where h_k denotes the output of the k'th layer of the discriminator d and d_k denotes the forward pass from layer k onwards. The first two terms comprise the ordinary GAN objective and the right-most term performs a convex combination between real and fake examples either in visible (pixel) space or hidden space (depending on the value of k).

It is worth noting that this specific formulation in the visible mixup case, i.e., when `k = 0`, is *not* equal to the formulation in the [original mixup paper](https://arxiv.org/pdf/1710.09412.pdf), since it did not work well in our experiments. This means that although Manifold Mixup is an exploration into performing mixup in the hidden space of the network, we do make a small contribution to visible mixup as well.

## Usage

### Requirements

* PyTorch 0.4 (or, at least a version that merges `Variable` and `Tensor`
* tqdm

### How to train

TODO

### How to evaluate

TODO: coming soon!

### Additional details

An experiment is defined within a bash script, such as the ones you see in the `exps` folder. Here is an example of one:

```
cd ..
BS=64
NAME=baseline_specnorm
python task_launcher.py \
--name=${NAME} \
--batch_size=${BS} \
--epochs=500 \
--z_dim=62 \
--beta1=0.0 \
--iterator=iterators/cifar10.py \
--network=networks/model_resnet_specnorm.py \
--save_every=20 \
--dnorm=0.0 \
--update_g_every=5 \
--resume=auto
```

In short, this runs an experiment without any mixup (since the `mixup=...` arg has not been defined) with latent dimension size 62 (`--z_dim=62`), a gradient norm penalty disabled (`--dnorm=0`), G is updated every five iterations (`--update_g_every=5`), and model saving every 10 epochs (`--save_every=10`). The `--resume` argument when set to `auto` means that when the script is run it will automatically find the latest saved model, load it, and resume the experiment. If no such models exist, it will run the experiment from scratch.

For more details, consult the code in `task_launcher.py`.

## Samples

![image](https://user-images.githubusercontent.com/2417792/41466513-99604f64-7070-11e8-96ce-4f747270d280.png) ![image](https://user-images.githubusercontent.com/2417792/41466524-ac1a81b0-7070-11e8-80ec-3711cfe3c209.png) ![image](https://user-images.githubusercontent.com/2417792/41466602-f5843c1a-7070-11e8-9b5e-a92f7adf7e83.png)

## Acknowledgements

This repo uses some code from:
* https://github.com/christiancosgrove/pytorch-spectral-normalization-gan

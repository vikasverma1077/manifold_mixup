# GAN mixup

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

The `exps` folder contains the experiments needed to reproduce the experiments in the paper. There is a non-trivial amount of variance in the Inception score between runs so we suggest you run each experiment at least thrice.

* `exps/baseline.specnorm.sh`: this is a strong baseline using the [spectral normalisation](https://arxiv.org/abs/1802.05957) proposed by Miyato et al (2018). After 500 epochs this achieved an average (over three runs) Inception score of 7.94 +/- 0.08, and an average FID of 21.7.
* `exps/baseline.specnorm.hinge.sh`: (another strong baseline) this is the best result from the spectral normalisation paper, where they were able to obtain a better Inception score by using the hinge loss. After 500 epochs we achieved 7.97 +/- 0.10 with an average FID of 22.2.
* `exps/visible.dnorm0.specnorm.sh`: this is visible (pixel-space) mixup using spectral norm. We obtained 8.00 +/- 0.08 with an FID of 21.5.
* `exps/hidden.dnorm0.5.pp.sh`: hidden space mixup using gradient norm penalty of 0.5 and using (as the hidden layer) a small convolution inserted before the resblocks of the discriminator.

### How to evaluate

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

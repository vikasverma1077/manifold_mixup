# Manifold_mixup
This repo consists Pytorch code for the paper Manifold Mixup: Better Representations by Interpolating Hidden States (https://arxiv.org/abs/1806.05236)

The goal of our proposed algorithm, Manifold Mixup, is to learn robust features by interpolating the hidden states of examples. The representations learned by our method are more discriminative and compact as shown in the below figure.  Please refer to Figure 1 and Figure 2 of our [paper](https://arxiv.org/abs/1806.05236) for more details.

<p align="center">
    <img src="mmfig1.png" height="600">
</p>

<p align="center">
    <img src="mmfig2.png" height="400">
</p>

The repo consist of three subfolders for Supervised Learning, Semi-Supervised Learning and GAN experiments. Each subfolder is self-contained (can be used independently of the other subfolders). Each subfolder has its own instruction on "How to run" in its README.md file.

If you find this work useful and use it on your own research, please cite our [paper](https://arxiv.org/abs/1806.05236). 

```
@article{manifold_mixup,
  title={Manifold Mixup: Encouraging Meaningful On-Manifold Interpolation as a Regularizer},
  author={Verma, Vikas and Lamb, Alex and Beckham, Christopher and Najafi, Amir and Courville, Aaron  and  Mitliagkis, Ioannis and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1806.05236v1},
  year={2018}
}
```


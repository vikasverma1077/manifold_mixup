# Manifold_mixup (ICML 2019)
This repo consists Pytorch code for the ICML 2019 paper Manifold Mixup: Better Representations by Interpolating Hidden States (https://arxiv.org/abs/1806.05236)

The goal of our proposed algorithm, Manifold Mixup, is to learn robust features by interpolating the hidden states of examples. The representations learned by our method are more discriminative and compact as shown in the below figure.  Please refer to Figure 1 and Figure 2 of our [paper](https://arxiv.org/abs/1806.05236) for more details.

<p align="center">
    <img src="mmfig1.png" height="600">
</p>

<p align="center">
    <img src="mmfig2.png" height="300">
</p>

The repo consist of two subfolders for Supervised Learning and GAN experiments. Each subfolder is self-contained (can be used independently of the other subfolders). Each subfolder has its own instruction on "How to run" in its README.md file.

If you find this work useful and use it on your own research, please cite our [paper](https://arxiv.org/abs/1806.05236). 

```
@ARTICLE{verma2018manifold,
       author = {{Verma}, Vikas and {Lamb}, Alex and {Beckham}, Christopher and
         {Najafi}, Amir and {Mitliagkas}, Ioannis and {Courville}, Aaron and
         {Lopez-Paz}, David and {Bengio}, Yoshua},
        title = "{Manifold Mixup: Better Representations by Interpolating Hidden States}",
      journal = {arXiv e-prints},
     keywords = {Statistics - Machine Learning, Computer Science - Artificial Intelligence, Computer Science - Machine Learning, Computer Science - Neural and Evolutionary Computing},
         year = "2018",
        month = "Jun",
          eid = {arXiv:1806.05236},
        pages = {arXiv:1806.05236},
archivePrefix = {arXiv},
       eprint = {1806.05236},
 primaryClass = {stat.ML},
       adsurl = {https://ui.adsabs.harvard.edu/\#abs/2018arXiv180605236V},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```
Note: Please refer to our new repo for Interpolation based Semi-supervised Learning https://github.com/vikasverma1077/ICT


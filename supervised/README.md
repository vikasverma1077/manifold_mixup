### Requirements
This code has been tested with  
python 2.7.9  
torch 0.3.1  
torchvision 0.2.0

### Data loading
Cifar 10 and Cifar100 data will be automatically downloaded in folder ../data/ if it does not exist there.

### How to run Supervised Manifold Mixup
```
python train_classifier_mixup.py --dataname cifar10 --epochs 200 --batch_size 100 --labels_per_class 5000 --mixup   --mixup_hidden --mixup_alpha 2.0 --learning_rate 0.1 --momentum 0.9 --schedule 100 150 --gammas 0.1 0.1  --model_type PreActResNet152 --fraction_validation 0.1 --exp_dir temp
```

### How to run Supervised Input Mixup
```
python train_classifier_mixup.py --dataname cifar10 --epochs 200 --batch_size 100 --labels_per_class 5000 --mixup   --mixup_alpha 1.0 --learning_rate 0.1 --momentum 0.9 --schedule 100 150 --gammas 0.1 0.1  --model_type PreActResNet152 --fraction_validation 0.1 --exp_dir temp
```


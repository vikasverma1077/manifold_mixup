
### How to run Semi-supervised experiments###


## For running Semi-supervised Manifold mixup for Cifar10

python main_mixup_hidden_ssl.py --optimizer sgd --lr 0.1 --epochs 1000 --batch_size 100  --mixup_sup 1 --mixup_usup 1  --mixup_u_u --net_type WRN28_2 --dataset cifar10 --schedule 500 750 875 --gammas 0.1 0.1 0.1  --alpha_max_at_factor 0.4  --alpha_max 1.0 --l2 0.0005 --nesterov --root_dir $HOME/experiments/SSL/ --data_dir ./data/cifar10/  --mixup_alpha_sup 0.1 --mixup_alpha_usup 2.0


## For running Semi-supervised Input mixup for Cifar10


python main_mixup_input_ssl.py --optimizer sgd --lr 0.1 --epochs 1000 --batch_size 100  --mixup_sup 1 --mixup_usup 1  --mixup_u_u --net_type WRN28_2 --dataset cifar10  --schedule 500 750 875 --gammas 0.1 0.1 0.1  --alpha_max_at_factor 0.4  --alpha_max 1.0 --l2 0.0005 --nesterov  --root_dir $HOME/experiments/WRN/ --data_dir $HOME/data/cifar10/ --mixup_alpha_sup 0.1 --mixup_alpha_usup 2.0

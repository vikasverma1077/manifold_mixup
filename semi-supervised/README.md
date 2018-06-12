
### How to run Semi-supervised experiments

Run the script python cifar10_zca.py to compute and save the ZCA for CIFAR10 in the folder ../data/cifar10/ 


## For running Semi-supervised Manifold mixup for Cifar10

python main_mixup_hidden_ssl.py  --dataset cifar10 --optimizer sgd --lr 0.1 --l2 0.0005 --nesterov --epochs 1000 --batch_size 100 --mixup_sup 1 --mixup_usup 1 --mixup_sup_hidden --mixup_usup_hidden  --mixup_alpha_sup 0.1 --mixup_alpha_usup 2.0 --alpha_max 1.0 --alpha_max_at_factor 0.4 --net_type WRN28_2 --schedule 500 750 875 --gammas 0.1 0.1 0.1 --exp_dir exp1 --data_dir ../data/cifar10/

## For running Semi-supervised Input mixup for Cifar10

python main_mixup_input_ssl.py  --dataset cifar10 --optimizer sgd --lr 0.1 --l2 0.0005 --nesterov --epochs 1000 --batch_size 100 --mixup_sup 1 --mixup_usup 1 --mixup_alpha_sup 0.1 --mixup_alpha_usup 2.0 --alpha_max 1.0 --alpha_max_at_factor 0.4 --net_type WRN28_2 --schedule 500 750 875 --gammas 0.1 0.1 0.1 --exp_dir exp2 --data_dir ../data/cifar10/


## For running Semi-supervised Manifold mixup for SVHN

python main_mixup_hidden_ssl.py  --dataset svhn --optimizer sgd --lr 0.1 --l2 0.0005 --nesterov --epochs 1000 --batch_size 100 --mixup_sup 1 --mixup_usup 1 --mixup_sup_hidden --mixup_usup_hidden  --mixup_alpha_sup 0.1 --mixup_alpha_usup 2.0 --alpha_max 2.0 --alpha_max_at_factor 0.4 --net_type WRN28_2 --schedule 500 750 875 --gammas 0.1 0.1 0.1 --exp_dir exp3 --data_dir ../data/svhn/

## For running Semi-supervised Input mixup for SVHN

python main_mixup_input_ssl.py  --dataset svhn --optimizer sgd --lr 0.1 --l2 0.0005 --nesterov --epochs 1000 --batch_size 100 --mixup_sup 1 --mixup_usup 1 --mixup_alpha_sup 0.1 --mixup_alpha_usup 2.0 --alpha_max 2.0 --alpha_max_at_factor 0.4 --net_type WRN28_2 --schedule 500 750 875 --gammas 0.1 0.1 0.1 --exp_dir exp4 --data_dir ../data/svhn/

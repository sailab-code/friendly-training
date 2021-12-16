### CNN-B on CIFAR10
## BASELINE (ratio_simp = 0.0)
python train-neural.py --activation=relu --arch=cnn2 --beta_simp=1000 --dataset=cifar10 --lr_clf=0.0001 --lr_simp=1e-05 --ratio_simp=0.0 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=no --epochs 200 --weight_decay_clf=0.0001 --weight_decay_simp=0.0001

## FT
python train-delta.py --arch=cnn2 --conf_thres=0.98 --dataset=cifar10 --epochs=200 --iterations_simp=10 --ratio_simp=0.25 --scaling=quadratic --step_simp=0.01

## NFT
python train-neural.py --activation=relu --arch=cnn2 --beta_simp=500 --dataset=cifar10 --epochs=200 --iterations_simp=1 --lr_clf=0.0001 --lr_simp=1e-05 --ratio_simp=0.85 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=no --weight_decay_clf=0.0001 --weight_decay_simp=0.0001

### RESNET on CIFAR10-N10
## BASELINE (ratio_simp = 0.0)
python train-neural.py --activation=relu --arch=resnet --beta_simp=2000 --dataset=cifar10-n10 --epochs=250 --iterations_simp=1 --lr_simp=0.0005 --ratio_simp=0 --run=4 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=no --weight_decay_simp=0.0001

## FT
python train-delta.py --arch=resnet --conf_thres=0.85 --dataset=cifar10-n10 --epochs=250 --iterations_simp=10 --ratio_simp=0.1 --scaling=quadratic --step_simp=10

## NFT
python train-neural.py --activation=relu --arch=resnet --beta_simp=1000 --dataset=cifar10-n10 --epochs=250 --iterations_simp=1 --lr_factor_clf=0.1 --lr_simp=0.0005 --n_deep=2 --n_filters_base=96 --noisy_labels=0.1 --ratio_simp=0.7 --run=3 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=no --weight_decay_simp=0.0001
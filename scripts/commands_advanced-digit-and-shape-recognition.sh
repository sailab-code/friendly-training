## NFT
python train-neural.py --activation=relu --arch=ff --beta_simp=2000 --dataset=mnist_back_image --iterations_simp=1 --lr_clf=0.0001 --lr_simp=0.0001 --n_deep=4 --n_filters_base=64 --ratio_simp=0.25 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=yes --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=ff --beta_simp=2000 --dataset=mnist_rot_back_image --iterations_simp=1 --lr_clf=0.0001 --lr_simp=1e-05 --n_deep=4 --n_filters_base=64 --ratio_simp=0.25 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=yes --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=ff --beta_simp=2000 --dataset=mnist_rot --iterations_simp=1 --lr_clf=0.0001 --lr_simp=1e-05 --n_deep=4 --n_filters_base=64 --ratio_simp=0.25 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=no --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=ff --beta_simp=500 --dataset=rectangles_image --iterations_simp=1 --lr_clf=0.0001 --lr_simp=1e-05 --n_deep=4 --n_filters_base=128 --ratio_simp=0.85 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=no --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=ff --beta_simp=2000 --dataset=convex --iterations_simp=1 --lr_clf=0.0001 --lr_simp=0.0005 --n_deep=4 --n_filters_base=128 --ratio_simp=0.85 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=no --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=ff2 --beta_simp=1000 --dataset=mnist_back_image --iterations_simp=1 --lr_clf=0.0001 --lr_simp=0.0001 --n_deep=4 --n_filters_base=96 --ratio_simp=0.5 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=no --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=ff2 --beta_simp=1000 --dataset=mnist_rot_back_image --iterations_simp=1 --lr_clf=0.0001 --lr_simp=1e-05 --n_deep=4 --n_filters_base=128 --ratio_simp=0.5 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=no --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=ff2 --beta_simp=500 --dataset=mnist_rot --iterations_simp=1 --lr_clf=0.0001 --lr_simp=0.0005 --n_deep=4 --n_filters_base=128 --ratio_simp=0.5 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=no --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=ff2 --beta_simp=1000 --dataset=rectangles_image --iterations_simp=1 --lr_clf=0.0001 --lr_simp=0.0001 --n_deep=4 --n_filters_base=64 --ratio_simp=0.5 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=no --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=ff2 --beta_simp=500 --dataset=convex --iterations_simp=1 --lr_clf=0.0001 --lr_simp=1e-05 --n_deep=4 --n_filters_base=128 --ratio_simp=0.25 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=no --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=cnn --beta_simp=500 --dataset=mnist_back_image --iterations_simp=1 --lr_clf=0.0001 --lr_simp=1e-05 --n_deep=4 --n_filters_base=96 --ratio_simp=0.85 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=no --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=cnn --beta_simp=2000 --dataset=mnist_rot_back_image --iterations_simp=1 --lr_clf=0.0001 --lr_simp=1e-05 --n_deep=4 --n_filters_base=96 --ratio_simp=0.25 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=no --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=cnn --beta_simp=1000 --dataset=mnist_rot --iterations_simp=1 --lr_clf=0.0001 --lr_simp=0.0005 --n_deep=4 --n_filters_base=96 --ratio_simp=0.25 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=no --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=cnn --beta_simp=500 --dataset=rectangles_image --iterations_simp=1 --lr_clf=0.0001 --lr_simp=0.0001 --n_deep=4 --n_filters_base=96 --ratio_simp=0.5 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=yes --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=cnn --beta_simp=2000 --dataset=convex --iterations_simp=1 --lr_clf=0.0001 --lr_simp=1e-05 --n_deep=4 --n_filters_base=128 --ratio_simp=0.25 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=yes --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=cnn2 --beta_simp=500 --dataset=mnist_back_image --iterations_simp=1 --lr_clf=0.0001 --lr_simp=1e-05 --n_deep=4 --n_filters_base=64 --ratio_simp=0.25 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=yes --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=cnn2 --beta_simp=2000 --dataset=mnist_rot_back_image --iterations_simp=1 --lr_clf=0.0001 --lr_simp=0.0005 --n_deep=4 --n_filters_base=64 --ratio_simp=0.25 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=yes --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=cnn2 --beta_simp=2000 --dataset=mnist_rot --iterations_simp=1 --lr_clf=0.0001 --lr_simp=0.0001 --n_deep=4 --n_filters_base=128 --ratio_simp=0.25 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=no --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=cnn2 --beta_simp=1000 --dataset=rectangles_image --iterations_simp=1 --lr_clf=0.0001 --lr_simp=0.0001 --n_deep=4 --n_filters_base=64 --ratio_simp=0.25 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=no --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=cnn2 --beta_simp=1000 --dataset=convex --iterations_simp=1 --lr_clf=0.0001 --lr_simp=0.0005 --n_deep=4 --n_filters_base=128 --ratio_simp=0.25 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=unet --target_conditioning=no --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
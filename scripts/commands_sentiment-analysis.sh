## BASELINES (ratio_simp = 0.0)
python train-neural.py --activation=relu --arch=ff2 --beta_simp=1000 --dataset=imdb50k --lr_clf=0.0001 --lr_simp=1e-05 --ratio_simp=0.0 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=ff --target_conditioning=no --epochs 30 --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=ff2 --beta_simp=1000 --dataset=winedr --lr_clf=0.0001 --lr_simp=1e-05 --ratio_simp=0.0 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=ff --target_conditioning=no --epochs 30 --weight_decay_clf=0.0001 --weight_decay_simp=0.0001

## FT
python train-delta.py --arch=ff2 --conf_thres=0.9 --dataset=imdb50k --epochs=30 --iterations_simp=120 --ratio_simp=0.25 --scaling=quadratic --step_simp=10
python train-delta.py --arch=ff2 --conf_thres=0.98 --dataset=winedr --epochs=30 --iterations_simp=80 --ratio_simp=0.5 --scaling=quadratic --step_simp=10

## NFT
python train-neural.py --activation=relu --arch=ff2 --beta_simp=1000 --dataset=imdb50k --epochs=30 --iterations_simp=1 --lr_clf=0.0001 --lr_simp=0.0001 --ratio_simp=0.5 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=ff --target_conditioning=yes --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
python train-neural.py --activation=relu --arch=ff2 --beta_simp=500 --dataset=winedr --epochs=30 --iterations_simp=1 --lr_clf=0.0001 --lr_simp=1e-05 --ratio_simp=0.5 --scaling=quadratic --sigmoid_postprocessing=no --simplifier=ff --target_conditioning=yes --weight_decay_clf=0.0001 --weight_decay_simp=0.0001
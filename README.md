# Anonymous submission to AAAI 2022
Title: **Being Friends Instead of Adversaries: Deep Networks Learn to Simplify Data to Train Other Networks**

_Notice that reproducibility is not guaranteed by PyTorch across different releases, platforms, hardware. Moreover, determinism cannot be enforced due to use of PyTorch operations for which deterministic implementations do not exist (e.g. bilinear upsampling)._

Make sure to have Python dependencies (including PyTorch) by running:
```
pip install -r requirements.txt
```

## Datasets
CIFAR-10 is publicly available (automatic download in `data` directory).

MNIST Variations and geometric shapes **[2]** are available at http://www.iro.umontreal.ca/~lisa/icml2007data (downloader script in `download_larochelle.sh`).

Wines reviews are available on [Kaggle](https://www.kaggle.com/zynicide/wine-reviews). The original data are included in `data` directory. Text vectorization should be performed by running the Python script `data/wines_data_processing.py`.

IMDB reviews are available at [stanford.edu](http://ai.stanford.edu/~amaas/data/sentiment/). Downloader script is available (see `download_imdb.sh`). Text vectorization should be performed by running the Python script `data/imdb_data_processing.py`.


## Launching experiments
Neural Friendly Training experiments can be launched with the `train-neural.py` Python script.

On the other hand, experiments with the FT approach (see Supplementary material and **[1]** for details) can be run with `train-delta.py`.

Example commands are available in the `scripts` directory (run them from the root directory). See `commands_advanced-digit-and-shape-recognition.sh`, `commands_image-classification.sh`, `commands_sentiment-analysis.sh` to repreduce results reported in the tables of the paper.

## Differences with the paper text
Algorithm 1 of paper text (NFT) is implemented in `train-neural.py` (optimization phases are swapped but equivalent).
Names of command line parameters slightly differ with respect to paper text, hence we report the name mapping (more details in the code).

Parameter $\frac{\gamma_{max\_{simp}}}{\gamma_{max}}$ is called `ratio_simp`, $\gamma_{max}$ corresponds to `epochs`.
Concerning the U-Net simplifier, $\nu$ is called `n_deep`, $n_f$ is `n_filters_base`.
$\alpha$, $\beta$ are termed `lr_clf`, `lr_simp`. $\eta_{max}$ is called `beta_simp`.

We implemented FT algorithm (see **[1]**) in `train-delta.py`.
$\frac{\xi_{max\_{simp}}}{\gamma_{max} |B|}$ is named `ratio_simp`, while $c$ is `conf_thres`.
$\alpha$ is referred to as `lr`, $\beta$ is `step_simp`.

Architectures FC-A, FC-B, CNN-A, CNN-B, ResNet18 are named `ff`, `ff2`, `cnn`, `cnn2`, `resnet`.

## References
**[1]** Marullo, S.; Tiezzi, M.; Gori, M.; and Melacci, S. 2021.
Friendly Training: Neural Networks Can Adapt Data To Make Learning Easier. 
In IEEE International Joint Conference on Neural Networks (IJCNN) (arXiv preprint arXiv:2106.10974).

**[2]** Larochelle, H.; Erhan, D.; Courville, A.; Bergstra, J.; and Bengio, Y. 2007.
An empirical evaluation of deep architectures on problems with many factors of variation. 
In International Conference on Machine Learning, 473–480.
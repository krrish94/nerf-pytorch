# nerf-pytorch

A PyTorch implementation of Neural Radiance Fields


## Tiny-NeRF on Google Colab

The NeRF code release has an accompanying Colab notebook, that showcases training a feature-limited version of NeRF on a "tiny" scene. It's equivalent PyTorch notebook can be found at the following URL:

https://colab.research.google.com/drive/1rO8xo0TemN67d4mTpakrKrLp03b9bgCX


## How to train your NeRF

To train a "full" NeRF model (i.e., using 3D coordinates as well as ray directions, and the hierarchical sampling procedure), first setup dependencies. In a new `conda` or `virtualenv` environment, run
```
pip install requirements.txt
```

**Importantly**, install [torchsearchsorted](https://github.com/aliutkus/torchsearchsorted) by following instructions from their `README`.

Once everything is setup, to run experiments, first edit `config/default.yml` to specify your own parameters.

The training script can be invoked by running
```
python train_nerf.py --config config/default.yml
```

Optionally, if resuming training from a previous checkpoint, run
```
python train_nerf.py --config config/default.yml --load-checkpoint path/to/checkpoint.ckpt
```

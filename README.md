# nerf-pytorch
#### A PyTorch re-implementation
### [Project](http://tancik.com/nerf) | [Video](https://youtu.be/JuH79E8rdKc) | [Paper](https://arxiv.org/abs/2003.08934)

[![Open Tiny-NeRF in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rO8xo0TemN67d4mTpakrKrLp03b9bgCX)

A PyTorch re-implementation of [Neural Radiance Fields](http://tancik.com/nerf).


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


## (Full) NeRF on Google Colab

A Colab notebook for the _full_ NeRF model (albeit on low-resolution data) can be accessed [here](https://colab.research.google.com/drive/1L6QExI2lw5xhJ-MLlIwpbgf7rxW7fcz3).


## A note on reproducibility

All said, this is not an official code release, and is instead a reproduction from the original code (released by the authors [here](https://github.com/bmild/nerf)).

I have currently ensured (to the best of my abilities, but feel free to open issues if you feel something's wrong :) ) that
* Every _individual_ module exactly (numerically) matches that of the TensorFlow implementation. [This Colab notebook](https://colab.research.google.com/drive/1ENrAtZIEhoeNkaXOXkBL7SbWU1VWHBQm) has all the tests, matching op for op (but is very scratchy to look at)!
* Training works as expected for fairly small resolutions (100 x 100).

However, this implementation still **lacks** the following:
* I have not run all the full experiments devised in the paper.
* I've only tested on the `lego` sequence of the synthetic (Blender) datasets.

The organization of code **WILL** change around a lot, because I'm actively experimenting with this.

**Pretrained models**: I am running a few large-scale experiments, and I hope to release models sometime in the end of April.


## Contributing / Issues?

Feel free to raise GitHub issues if you find anything concerning. Pull requests adding additional features are welcome too.


## LICENSE

`nerf-pytorch` is available under the [MIT License](https://opensource.org/licenses/MIT). For more details see: [LICENSE](LICENSE) and [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS).

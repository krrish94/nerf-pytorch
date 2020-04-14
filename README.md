# nerf-pytorch
#### A PyTorch re-implementation
### [Project](http://tancik.com/nerf) | [Video](https://youtu.be/JuH79E8rdKc) | [Paper](https://arxiv.org/abs/2003.08934)

[![Open Tiny-NeRF in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rO8xo0TemN67d4mTpakrKrLp03b9bgCX)

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://tancik.com/nerf)  
 [Ben Mildenhall](https://people.eecs.berkeley.edu/~bmild/)\*<sup>1</sup>,
 [Pratul P. Srinivasan](https://people.eecs.berkeley.edu/~pratul/)\*<sup>1</sup>,
 [Matthew Tancik](http://tancik.com/)\*<sup>1</sup>,
 [Jonathan T. Barron](http://jonbarron.info/)<sup>2</sup>,
 [Ravi Ramamoorthi](http://cseweb.ucsd.edu/~ravir/)<sup>3</sup>,
 [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html)<sup>1</sup> <br>
 <sup>1</sup>UC Berkeley, <sup>2</sup>Google Research, <sup>3</sup>UC San Diego  
  \*denotes equal contribution

<p align="center">
    <img src="assets/pipeline.jpg"/>
</p>

A PyTorch re-implementation of [Neural Radiance Fields](http://tancik.com/nerf).

The current implementation is blazingly fast! (Thorough benchmark to come, but **~2x faster**)


## Tiny-NeRF on Google Colab

The NeRF code release has an accompanying Colab notebook, that showcases training a feature-limited version of NeRF on a "tiny" scene. It's equivalent PyTorch notebook can be found at the following URL:

https://colab.research.google.com/drive/1rO8xo0TemN67d4mTpakrKrLp03b9bgCX


## What is a NeRF?

A neural radiance field is a simple fully connected network (weights are ~5MB) trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new views.

Optimizing a NeRF takes between a few hours and a day or two (depending on resolution) and only requires a single GPU. Rendering an image from an optimized NeRF takes somewhere between less than a second and ~30 seconds, again depending on resolution.


## How to train your NeRF

To train a "full" NeRF model (i.e., using 3D coordinates as well as ray directions, and the hierarchical sampling procedure), first setup dependencies. In a new `conda` or `virtualenv` environment, run
```bash
pip install requirements.txt
```

**Importantly**, install [torchsearchsorted](https://github.com/aliutkus/torchsearchsorted) by following instructions from their `README`.

Once everything is setup, to run experiments, first edit `config/lego.yml` to specify your own parameters.

The training script can be invoked by running
```bash
python train_nerf.py --config config/lego.yml
```

Optionally, if resuming training from a previous checkpoint, run
```bash
python train_nerf.py --config config/lego.yml --load-checkpoint path/to/checkpoint.ckpt
```


## (Full) NeRF on Google Colab

A Colab notebook for the _full_ NeRF model (albeit on low-resolution data) can be accessed [here](https://colab.research.google.com/drive/1L6QExI2lw5xhJ-MLlIwpbgf7rxW7fcz3).


## Render fun videos

Once you've trained your NeRF, it's time to use that to render the scene. Use the `eval_nerf.py` script to do that.
```bash
python eval_nerf.py --config logs/experiment_id/config.yml --checkpoint logs/experiment_id/checkpoint100000.ckpt --savedir cache/rendered/experiment_id
```

You can create a `gif` out of the saved images, for instance, by using [Imagemagick](https://imagemagick.org/).
```bash
convert cache/rendered/experiment_id/*.png cache/rendered/experiment_id.gif
```


## A note on reproducibility

All said, this is not an official code release, and is instead a reproduction from the original code (released by the authors [here](https://github.com/bmild/nerf)).

The code is thoroughly tested (to the best of my abilities) to match the original implementation (and be much faster)! In particular, I have ensured that
* Every _individual_ module exactly (numerically) matches that of the TensorFlow implementation. [This Colab notebook](https://colab.research.google.com/drive/1ENrAtZIEhoeNkaXOXkBL7SbWU1VWHBQm) has all the tests, matching op for op (but is very scratchy to look at)!
* Training works as expected (for Lego and LLFF scenes).

The organization of code **WILL** change around a lot, because I'm actively experimenting with this.

**Pretrained models**: I am running a few large-scale experiments, and I hope to release models sometime in the end of April.


## Contributing / Issues?

Feel free to raise GitHub issues if you find anything concerning. Pull requests adding additional features are welcome too.


## LICENSE

`nerf-pytorch` is available under the [MIT License](https://opensource.org/licenses/MIT). For more details see: [LICENSE](LICENSE) and [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS).

## Misc

Also, a shoutout to [yenchenlin](https://github.com/yenchenlin) for his cool PyTorch [implementation](https://github.com/yenchenlin/nerf-pytorch), whose volume rendering function replaced mine (my initial impl was inefficient in comparison).

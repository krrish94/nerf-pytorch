import argparse
import os
import yaml

import numpy as np
import torch

import models
from cfgnode import CfgNode
from load_blender import load_blender_data
from nerf_helpers import get_minibatches, get_ray_bundle
from nerf_helpers import meshgrid_xy
from train_utils import predict_and_render_radiance
from train_utils import run_network, run_one_iter_of_nerf


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    torch.autograd.set_detect_anomaly(True)

    images, poses, render_poses, hwf, i_split = load_blender_data(
        cfg.dataset.basedir,
        half_res=cfg.dataset.half_res,
        testskip=cfg.dataset.testskip
    )
    i_train, i_val, i_test = i_split
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Initialize a coarse-resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)()
    model_coarse.to(device)
    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)()
        model_fine.to(device)

    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    if model_fine is not None:
        trainable_parameters += list(model_fine.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )

    # render_kwargs_train = {
    #     'perturb' : 1.,
    #     'N_importance' : 64,
    #     'network_fine' : model_fine,
    #     'embed_fn' : positional_encoding,
    #     'N_samples' : 64,
    #     'network_coarse' : model_coarse,
    #     'use_viewdirs' : True,
    #     'embeddirs_fn' : positional_encoding,
    #     'white_background' : False,
    #     'radiance_field_noise_std': 0.,
    #     'near': 0.,
    #     'far': 1.,
    #     'ndc': True,
    #     'lindisp': False,
    # }

    # # TODO: Prepare raybatch tensor if batching random rays

    for i in range(1000):
        img_idx = np.random.choice(i_train)
        img_target = images[img_idx].to(device)
        pose_target = poses[img_idx, :3, :4].to(device)
        ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
        coords = torch.stack(meshgrid_xy(torch.arange(H).to(device),
                                         torch.arange(W).to(device)), dim=-1)
        coords = coords.reshape((-1, 2))
        select_inds = np.random.choice(
            coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False
        )
        select_inds = coords[select_inds]
        ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
        ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
        batch_rays = torch.stack([ray_origins, ray_directions], dim=0)
        target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]
        
        rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine = run_one_iter_of_nerf(
            H, W, focal, model_coarse, model_fine, batch_rays, cfg, mode="train"
        )

        coarse_loss = torch.nn.functional.mse_loss(
            rgb_coarse[..., :3], target_s[..., :3]
        )
        fine_loss = None
        if rgb_fine is not None:
            fine_loss = torch.nn.functional.mse_loss(
                rgb_fine[..., :3], target_s[..., :3]
            )
        # loss = torch.nn.functional.mse_loss(rgb_pred[..., :3], target_s[..., :3])
        loss = coarse_loss + (fine_loss if fine_loss is not None else 0.)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Loss:", loss.item())

    print("Done!")

if __name__ == "__main__":
    main()

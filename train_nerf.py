import argparse
import os
import time
import yaml

import glob
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import models
from cfgnode import CfgNode
from load_blender import load_blender_data
from metrics import ScalarMetric
from nerf_helpers import get_minibatches, get_ray_bundle
from nerf_helpers import img2mse, meshgrid_xy, mse2psnr
from train_utils import predict_and_render_radiance
from train_utils import eval_nerf
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

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # If a pre-cached dataset is available, skip the dataloader.
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split = None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    if os.path.exists(cfg.dataset.cachedir):
        train_paths = glob.glob(
            os.path.join(cfg.dataset.cachedir, "train", "*.data")
        )
        validation_paths = glob.glob(
            os.path.join(cfg.dataset.cachedir, "val", "*.data")
        )
    else:
        # Load dataset
        images, poses, render_poses, hwf, i_split = load_blender_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip
        )
        i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
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

    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    # Write out config parameters.
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump()) #cfg, f, default_flow_style=False)

    # # TODO: Prepare raybatch tensor if batching random rays

    for i in trange(cfg.experiment.train_iters):

        model_coarse.train()
        if model_fine:
            model_coarse.train()

        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        if cfg.dataset.cachedir:
            datafile = np.random.choice(train_paths)
            cache_dict = torch.load(datafile)
            rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                cache_dict["height"], cache_dict["width"], cache_dict["focal_length"],
                model_coarse, model_fine, cache_dict["ray_bundle"].to(device), cfg,
                mode="train"
            )
            target_ray_values = cache_dict["target"].to(device)
        else:
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
            
            rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                H, W, focal, model_coarse, model_fine, batch_rays, cfg, mode="train"
            )
            target_ray_values = target_s

        coarse_loss = torch.nn.functional.mse_loss(
            rgb_coarse[..., :3], target_ray_values[..., :3]
        )
        fine_loss = None
        if rgb_fine is not None:
            fine_loss = torch.nn.functional.mse_loss(
                rgb_fine[..., :3], target_ray_values[..., :3]
            )
        # loss = torch.nn.functional.mse_loss(rgb_pred[..., :3], target_s[..., :3])
        loss = coarse_loss + (fine_loss if fine_loss is not None else 0.)
        loss.backward()
        psnr = mse2psnr(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write("[TRAIN] Iter: " + str(i) + " Loss: " + str(loss.item()) + " PSNR: " + str(psnr))
        writer.add_scalar("train/loss", loss.item(), i)
        writer.add_scalar("train/psnr", psnr, i)

        # Validation
        if i % cfg.experiment.validate_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write("[VAL] =======> Iter: " + str(i))
            model_coarse.eval()
            if model_fine:
                model_coarse.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None
                if cfg.dataset.cachedir:
                    datafile = np.random.choice(validation_paths)
                    cache_dict = torch.load(datafile)
                    rgb_coarse, _, _, rgb_fine, _, _ = eval_nerf(
                        cache_dict["height"], cache_dict["width"], cache_dict["focal_length"],
                        model_coarse, model_fine, cache_dict["ray_origins"].to(device),
                        cache_dict["ray_directions"].to(device), cfg, mode="train"
                    )
                    target_ray_values = cache_dict["target"].to(device)
                else:
                    img_idx = np.random.choice(i_val)
                    img_target = images[img_idx].to(device)
                    pose_target = poses[img_idx, :3, :4].to(device)
                    ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
                    rgb_coarse, _, _, rgb_fine, _, _ = eval_nerf(
                        H, W, focal, model_coarse, model_fine, ray_origins, ray_directions, cfg
                    )
                    target_ray_values = img_target
                coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                fine_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                loss = coarse_loss + fine_loss
                psnr = mse2psnr(loss.item())
                writer.add_scalar("validation/loss", loss.item(), i)
                writer.add_scalar("validataion/psnr", psnr, i)
                writer.add_image("validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]).detach().cpu())
                writer.add_image("validation/rgb_fine", cast_to_image(rgb_fine[..., :3]).detach().cpu())
                writer.add_image("validation/img_target", cast_to_image(target_ray_values[..., :3]).detach().cpu())
                tqdm.write("Validation loss: " + str(loss.item()) + " Validation PSNR: " + str(psnr) + "Time: " + str(time.time() - start))

        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            torch.save({
                "iter": i,
                "model_coarse_state_dict": model_coarse.state_dict(),
                "model_fine_state_dict": model_fine.state_dict(),
                "loss": loss,
                "psnr": psnr
            }, os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"))
            tqdm.write("================== Saved Checkpoint =================")


    print("Done!")


def cast_to_image(tensor):
    if tensor.min() != tensor.max():
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = (tensor * 255).clamp(0., 255.)
    tensor = tensor.long()
    # Flip to channels first.
    tensor = tensor.permute(2, 0, 1)
    return tensor


if __name__ == "__main__":
    main()

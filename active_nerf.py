"""
Replicates the train_nerf.py script, but does active learning
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import torchvision
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import models
from cfgnode import CfgNode
from lieutils import SE3
from load_blender import load_blender_data
from nerf_helpers import (
    get_ray_bundle,
    img2mse,
    meshgrid_xy,
    mse2psnr,
    positional_encoding,
)
from train_utils import eval_nerf, run_one_iter_of_nerf


def compute_distmat_se3(poses, num_train):
    posedists = torch.zeros(num_train, num_train)
    for i in range(num_train):
        for j in range(num_train):
            posedists[i, j] = SE3.Log(torch.matmul(poses[i].inverse(), poses[j])).norm()
            posedists[j, i] = SE3.Log(torch.matmul(poses[j].inverse(), poses[i])).norm()
    return posedists


def farthest_sequence_sampling(posedists):
    r"""https://gist.github.com/ctralie/128cc07da67f1d2e10ea470ee2d23fe8
    """
    seqlen = posedists.shape[0]
    ds = posedists[0]
    farthest_sequence = torch.zeros(seqlen).long()
    sampling_lambdas = torch.zeros(seqlen)
    for i in range(seqlen):
        tidx = np.argmax(ds)
        farthest_sequence[i] = tidx
        sampling_lambdas[i] = ds[tidx]
        ds = np.minimum(ds, posedists[tidx, :])
    return farthest_sequence


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    parser.add_argument(
        "--passive",
        action="store_true",
        help="Emulate a passive training strategy (i.e., don't use active steps).",
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
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split = None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    if hasattr(cfg.dataset, "cachedir") and os.path.exists(cfg.dataset.cachedir):
        train_paths = glob.glob(os.path.join(cfg.dataset.cachedir, "train", "*.data"))
        validation_paths = glob.glob(
            os.path.join(cfg.dataset.cachedir, "val", "*.data")
        )
        USE_CACHED_DATASET = True
    else:
        # Load dataset
        images, poses, render_poses, hwf, i_split = load_blender_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
        )
        i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

    # Tensor to hold indices of negative examples. Initially, every ray is a negative example.
    print("Initializing negative example stores.")
    negative_examples = torch.ones(
        len(i_train), images[0].shape[0], images[0].shape[1]
    ).bool()
    negative_examples = negative_examples.reshape(len(i_train), -1)
    print("Computing pairwise distances among poses.")
    posedists = compute_distmat_se3(poses, len(i_train))
    print("Farthest sequence sampling among poses.")
    farthest_sequence = farthest_sequence_sampling(posedists)
    next_img_idx = farthest_sequence[0]
    reset_every = 500

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # # Encoding function for position (xyz).
    # encode_position_fn = lambda x: positional_encoding(
    #     x, num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
    #     include_input=cfg.models.coarse.include_input_xyz
    # )
    # # Encoding function for direction.
    # encode_direction_fn = lambda x: positional_encoding(
    #     x, num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
    #     include_input=cfg.models.coarse.include_input_dir
    # )

    def encode_position_fn(x):
        return positional_encoding(
            x,
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
            include_input=cfg.models.coarse.include_input_xyz,
        )

    def encode_direction_fn(x):
        return positional_encoding(
            x,
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
        )

    # Initialize a coarse-resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
    )
    model_coarse.to(device)
    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
        )
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
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # # TODO: Prepare raybatch tensor if batching random rays

    for i in trange(cfg.experiment.train_iters):

        model_coarse.train()
        if model_fine:
            model_coarse.train()

        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        if USE_CACHED_DATASET:
            datafile = np.random.choice(train_paths)
            cache_dict = torch.load(datafile)
            ray_bundle = cache_dict["ray_bundle"]
            ray_origins, ray_directions = (
                ray_bundle[0].reshape((-1, 3)),
                ray_bundle[1].reshape((-1, 3)),
            )
            target_ray_values = cache_dict["target"][..., :3].reshape((-1, 3))
            select_inds = np.random.choice(
                ray_origins.shape[0],
                size=(cfg.nerf.train.num_random_rays),
                replace=False,
            )
            ray_origins, ray_directions = (
                ray_origins[select_inds],
                ray_directions[select_inds],
            )
            target_ray_values = target_ray_values[select_inds].to(device)
            ray_bundle = torch.stack([ray_origins, ray_directions], dim=0).to(device)
            rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                cache_dict["height"],
                cache_dict["width"],
                cache_dict["focal_length"],
                model_coarse,
                model_fine,
                ray_bundle,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )
        else:
            # img_idx = np.random.choice(i_train)
            img_idx = next_img_idx
            img_target = images[img_idx].to(device)
            pose_target = poses[img_idx, :3, :4].to(device)
            ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
            coords = torch.stack(
                meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
                dim=-1,
            )
            coords = coords.reshape((-1, 2))
            # select_inds = np.random.choice(
            #     coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False
            # )
            randperm, num_select = None, None
            if configargs.passive is True:
                randperm = torch.randperm(coords.shape[0])
                num_select = min(cfg.nerf.train.num_random_rays, coords.shape[0])
            else:
                active_negative_inds = torch.where(negative_examples[img_idx])
                active_negative_inds = active_negative_inds[0]
                randperm = torch.randperm(active_negative_inds.shape[0])
                num_select = min(
                    cfg.nerf.train.num_random_rays, active_negative_inds.shape[0]
                )
            select_inds_ = randperm[:num_select]
            select_inds = coords[select_inds_]
            ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
            ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
            batch_rays = torch.stack([ray_origins, ray_directions], dim=0)
            target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]

            rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                H,
                W,
                focal,
                model_coarse,
                model_fine,
                batch_rays,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
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
        loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
        loss.backward()
        psnr = mse2psnr(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write(
                "[TRAIN] Iter: "
                + str(i)
                + " Loss: "
                + str(loss.item())
                + " PSNR: "
                + str(psnr)
            )
        writer.add_scalar("train/loss", loss.item(), i)
        writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
        if rgb_fine is not None:
            writer.add_scalar("train/fine_loss", fine_loss.item(), i)
        writer.add_scalar("train/psnr", psnr, i)

        # Per-ray errors
        errs = (rgb_coarse[..., :3] - target_ray_values[..., :3]).norm(p=2, dim=-1)
        # Update active set
        tol = 1e-3
        negative_examples[img_idx, select_inds_[errs < tol]] = False

        next_img_idx = farthest_sequence[i % len(farthest_sequence)]
        if i % reset_every == 0:
            negative_examples = torch.ones(
                len(i_train), images[0].shape[0], images[0].shape[1]
            ).bool()
            negative_examples = negative_examples.reshape(len(i_train), -1)

        # Validation
        if (
            i % cfg.experiment.validate_every == 0
            or i == cfg.experiment.train_iters - 1
        ):
            tqdm.write("[VAL] =======> Iter: " + str(i))
            model_coarse.eval()
            if model_fine:
                model_coarse.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None
                if USE_CACHED_DATASET:
                    datafile = np.random.choice(validation_paths)
                    cache_dict = torch.load(datafile)
                    rgb_coarse, _, _, rgb_fine, _, _ = eval_nerf(
                        cache_dict["height"],
                        cache_dict["width"],
                        cache_dict["focal_length"],
                        model_coarse,
                        model_fine,
                        cache_dict["ray_origins"].to(device),
                        cache_dict["ray_directions"].to(device),
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = cache_dict["target"].to(device)
                else:
                    img_idx = np.random.choice(i_val)
                    img_target = images[img_idx].to(device)
                    pose_target = poses[img_idx, :3, :4].to(device)
                    ray_origins, ray_directions = get_ray_bundle(
                        H, W, focal, pose_target
                    )
                    rgb_coarse, _, _, rgb_fine, _, _ = eval_nerf(
                        H,
                        W,
                        focal,
                        model_coarse,
                        model_fine,
                        ray_origins,
                        ray_directions,
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = img_target
                coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                fine_loss = 0.0
                if rgb_fine is not None:
                    fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                loss = coarse_loss + fine_loss
                psnr = mse2psnr(loss.item())
                writer.add_scalar("validation/loss", loss.item(), i)
                writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                writer.add_scalar("validataion/psnr", psnr, i)
                writer.add_image(
                    "validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3])
                )
                if rgb_fine is not None:
                    writer.add_image(
                        "validation/rgb_fine", cast_to_image(rgb_fine[..., :3])
                    )
                    writer.add_scalar("validation/fine_loss", fine_loss.item(), i)
                writer.add_image(
                    "validation/img_target", cast_to_image(target_ray_values[..., :3])
                )
                tqdm.write(
                    "Validation loss: "
                    + str(loss.item())
                    + " Validation PSNR: "
                    + str(psnr)
                    + "Time: "
                    + str(time.time() - start)
                )

        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": model_coarse.state_dict(),
                "model_fine_state_dict": None
                if not model_fine
                else model_fine.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "psnr": psnr,
            }
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")

    print("Done!")


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img


if __name__ == "__main__":
    main()

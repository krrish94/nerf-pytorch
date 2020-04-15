"""
Script to run and cache a dataset (ray bundles and ground-truth values)
for faster train-eval loops.
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from nerf import get_ray_bundle, load_blender_data, load_llff_data, meshgrid_xy


def cache_nerf_dataset(args):

    images, poses, render_poses, hwf = (
        None,
        None,
        None,
        None,
    )
    i_train, i_val, i_test = None, None, None

    if args.type == "blender":
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datapath, half_res=args.blender_half_res, testskip=args.blender_stride
        )

        i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
    elif args.type == "llff":
        images, poses, bds, render_poses, i_test = load_llff_data(
            args.datapath, factor=args.llff_downsample_factor
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        if not isinstance(i_test, list):
            i_test = [i_test]
        if args.llffhold > 0:
            i_test = np.arange(images.shape[0])[:: args.llffhold]
        i_val = i_test
        i_train = np.array(
            [
                i
                for i in np.arange(images.shape[0])
                if (i not in i_test and i not in i_val)
            ]
        )
        H, W, focal = hwf
        hwf = [int(H), int(W), focal]
        images = torch.from_numpy(images)
        poses = torch.from_numpy(poses)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    os.makedirs(os.path.join(args.savedir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.savedir, "val"), exist_ok=True)
    os.makedirs(os.path.join(args.savedir, "test"), exist_ok=True)
    np.random.seed(args.randomseed)

    for img_idx in tqdm(i_train):
        for j in range(args.num_variations):
            img_target = images[img_idx].to(device)
            pose_target = poses[img_idx, :3, :4].to(device)
            ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
            coords = torch.stack(
                meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
                dim=-1,
            )
            coords = coords.reshape((-1, 2))
            target_s = None
            save_path = None
            if args.sample_all is False:
                select_inds = np.random.choice(
                    coords.shape[0], size=(args.num_random_rays), replace=False
                )
                select_inds = coords[select_inds]
                ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
                ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
                target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]
                save_path = os.path.join(
                    args.savedir,
                    "train",
                    str(img_idx).zfill(4),
                    str(j).zfill(4),
                    ".data",
                )
            else:
                target_s = img_target
                save_path = os.path.join(
                    args.savedir, "train", str(img_idx).zfill(4) + ".data"
                )

            batch_rays = torch.stack([ray_origins, ray_directions], dim=0)

            cache_dict = {
                "height": H,
                "width": W,
                "focal_length": focal,
                "ray_bundle": batch_rays.detach().cpu(),
                "target": target_s.detach().cpu(),
            }

            save_path = os.path.join(
                args.savedir, "train", str(img_idx).zfill(4) + ".data"
            )
            torch.save(cache_dict, save_path)

            if args.sample_all is True:
                break

    for img_idx in tqdm(i_val):
        img_target = images[img_idx].to(device)
        pose_target = poses[img_idx, :3, :4].to(device)
        ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)

        cache_dict = {
            "height": H,
            "width": W,
            "focal_length": focal,
            "ray_origins": ray_origins.detach().cpu(),
            "ray_directions": ray_directions.detach().cpu(),
            "target": img_target.detach().cpu(),
        }

        save_path = os.path.join(args.savedir, "val", str(img_idx).zfill(4) + ".data")
        torch.save(cache_dict, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="Path to root dir of dataset that needs caching.",
    )
    parser.add_argument(
        "--type",
        type=str.lower,
        required=True,
        choices=["blender", "llff"],
        help="Type of the dataset to be cached.",
    )
    parser.add_argument(
        "--blender-half-res",
        type=bool,
        default=True,
        help="Whether to load the (Blender/synthetic) datasets"
        "at half the resolution.",
    )
    parser.add_argument(
        "--blender-stride",
        type=int,
        default=1,
        help="Stride length (Blender datasets only). When set to k (k > 1), it samples"
        "every kth sample from the dataset.",
    )
    parser.add_argument(
        "--llff-downsample-factor",
        type=int,
        default=8,
        help="Downsample factor for images from the LLFF dataset.",
    )
    parser.add_argument(
        "--llffhold",
        type=int,
        default=8,
        help="Determines the hold-out images for LLFF (TODO: make better).",
    )
    parser.add_argument(
        "--savedir", type=str, required=True, help="Path to save the cached dataset to."
    )
    parser.add_argument(
        "--num-random-rays",
        type=int,
        default=8,
        help="Number of random rays to sample per image.",
    )
    parser.add_argument(
        "--num-variations",
        type=int,
        default=1,
        help="Number of random 'ray batches' to draw per image",
    )
    parser.add_argument(
        "--sample-all",
        action="store_true",
        help="Sample all rays for the image. Overrides --num-random-rays.",
    )
    parser.add_argument(
        "--randomseed", type=int, default=3920, help="Random seeed, for repeatability"
    )
    args = parser.parse_args()

    cache_nerf_dataset(args)

import copy
import os

import numpy as np
import torch

from load_blender import load_blender_data
from nerf_helpers import cumprod_exclusive
from nerf_helpers import get_minibatches, get_ray_bundle
from nerf_helpers import meshgrid_xy, positional_encoding
from nerf_helpers import ndc_rays, sample_pdf


class VeryTinyNerfModel(torch.nn.Module):
    r"""Define a "very tiny" NeRF model comprising three fully connected layers.
    """

    def __init__(self, filter_size=128, num_encoding_functions=6, use_viewdirs=True):
        super(VeryTinyNerfModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        # Input layer (default: 65 -> 128)
        self.layer1 = torch.nn.Linear(self.xyz_encoding_dims + self.viewdir_encoding_dims, filter_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        # Layer 3 (default: 128 -> 4)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


def render_volume_density(radiance_field, depth_values, rays_d, options):
    # TESTED
    one_e_10 = torch.tensor([1e10], dtype=rays_d.dtype, device=rays_d.device)
    dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1],
                       one_e_10.expand(depth_values[..., :1].shape)), dim=-1)
    dists = dists * rays_d[..., None, :].norm(p=2, dim=-1)

    rgb = torch.sigmoid(radiance_field[..., :3])
    noise = 0.
    if options["raw_noise_std"] > 0.:
        noise = torch.randn(radiance_field[..., 3].shape) * options["raw_noise_std"]
        noise = noise.to(radiance_field)
    sigma_a = torch.nn.functional.relu(radiance_field[..., 3] + noise)
    alpha = 1. - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)
    depth_map = weights * depth_values
    depth_map = depth_map.sum(dim=-1)
    # depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    disp_map = 1. / torch.max(
        1e-10 * torch.ones_like(depth_map), depth_map / acc_map
    )

    if options["white_bkgd"]:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def run_network(pts, network_fn, ray_batch, chunksize, options):

    pts_flat = pts.reshape((-1, pts.shape[-1]))
    embedded = options["embed_fn"](pts_flat)
    if options["embeddirs_fn"] is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = options["embeddirs_fn"](input_dirs_flat)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)

    batches = get_minibatches(embedded, chunksize=chunksize)
    preds = []
    for batch in batches:
        preds.append(network_fn(batch))
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(list(pts.shape[:-1]) + [radiance_field.shape[-1]])
    return radiance_field


def render_rays(ray_batch, chunksize, options):
    # TESTED
    N_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    bounds = ray_batch[..., 6:8].reshape((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    # TODO: Use actual values for "near" and "far" (instead of 0. and 1.)
    # when not enabling "ndc".
    t_vals = torch.linspace(0., 1., options["N_samples"]).to(ro)
    if not options["lindisp"]:
        z_vals = near * (1. - t_vals) + far * t_vals
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    z_vals = z_vals.expand([N_rays, options["N_samples"]])

    if options  ["perturb"] > 0.:
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(z_vals.shape).to(ro)
        z_vals = lower + (upper - lower) * t_rand
    # pts -> (N_rays, N_samples, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    radiance_field = run_network(pts, options["network_coarse"], ray_batch,
                                 chunksize, options)
    
    rgb_coarse, disp_coarse, acc_coarse, weights, depth_coarse = render_volume_density(
        radiance_field, z_vals, rd, options
    )

    # TODO: Implement importance sampling, and finer network.
    rgb_fine, disp_fine, acc_fine = None, None, None
    if options["N_importance"] > 0:
        # rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], options["N_importance"],
            det=(options["perturb"] == 0.)
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
        # pts -> (N_rays, N_samples + N_importance, 3)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

        radiance_field = run_network(pts, options["network_fine"], ray_batch,
                                     chunksize, options)
        rgb_fine, disp_fine, acc_fine, _, _ = render_volume_density(
            radiance_field, z_vals, rd, options
        )

    # retvals = {}
    # retvals["rgb_fine"] = rgb_map
    # retvals["disp_fine"] = disp_map
    # retvals["acc_fine"] = acc_map
    # # retvals["radiance_field"] = radiance_field

    # if options["N_importance"] > 0:
    #     retvals["rgb_coarse"] = rgb_map_0
    #     retvals["disp_coarse"] = disp_map_0
    #     retvals["acc_map"] = acc_map_0
    #     retvals["z_std"] = z_samples.std(dim=-1)

    return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
    # return rgb_map, disp_map, acc_map, radiance_field


def run_one_iter_of_nerf(H, W, focal, chunksize, batch_rays, options):
    ray_origins = batch_rays[0]
    ray_directions = batch_rays[1]
    if options['use_viewdirs']:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.reshape((-1, 3))
    ray_shapes = ray_directions.shape  # Cache now, to restore later.
    ro, rd = ndc_rays(H, W, focal, 1., ray_origins, ray_directions)
    ro = ro.reshape((-1, 3))
    rd = rd.reshape((-1, 3))
    near = options["near"] * torch.ones_like(rd[..., :1])
    far = options["far"] * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)
    if options['use_viewdirs']:
        rays = torch.cat((rays, viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=chunksize)
    # TODO: Init a list, keep appending outputs to that list,
    # concat everything in the end.
    rgb_coarse, disp_coarse, acc_coarse = [], [], []
    rgb_fine, disp_fine, acc_fine = None, None, None
    for batch in batches:
        rc, dc, ac, rf, df, af = render_rays(batch, chunksize, options)
        rgb_coarse.append(rc)
        disp_coarse.append(dc)
        acc_coarse.append(ac)
        if rf is not None:
            if rgb_fine is None:
                rgb_fine = [rf]
            else:
                rgb_fine.append(rf)
        if df is not None:
            if disp_fine is None:
                disp_fine = [df]
            else:
                disp_fine.append(df)
        if af is not None:
            if acc_fine is None:
                acc_fine = [af]
            else:
                acc_fine.append(af)
        # rgb_map.append(rgb_map_)
        # disp_map.append(disp_map_)
        # acc_map.append(acc_map_)
        # radiance_field.append(radiance_field_)

    # rgb_map = torch.cat(rgb_map, dim=0)
    # disp_map = torch.cat(disp_map, dim=0)
    # acc_map = torch.cat(acc_map, dim=0)
    # radiance_field = torch.cat(radiance_field, dim=0)

    rgb_coarse_ = torch.cat(rgb_coarse, dim=0)
    disp_coarse_ = torch.cat(disp_coarse, dim=0)
    acc_coarse_ = torch.cat(acc_coarse, dim=0)
    if rgb_fine is not None:
        rgb_fine_ = torch.cat(rgb_fine, dim=0)
    else:
        rgb_fine_ = None
    if disp_fine is not None:
        disp_fine_ = torch.cat(disp_fine, dim=0)
    else:
        disp_fine_ = None
    if acc_fine is not None:
        acc_fine_ = torch.cat(acc_fine, dim=0)
    else:
        acc_fine_ = None

    # return returndict
    return rgb_coarse_, disp_coarse_, acc_coarse_, rgb_fine_, disp_fine_, acc_fine_
    # return rgb_map, disp_map, acc_map, radiance_field


def main():

    torch.autograd.set_detect_anomaly(True)

    images, poses, render_poses, hwf, i_split = load_blender_data(
        "cache/nerf_example_data/nerf_synthetic/lego",
        half_res=True,
        testskip=50
    )
    i_train, i_val, i_test = i_split
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    N_rand = 32  # 32 * 32 * 4
    chunksize = 8

    seed = 8239
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda"

    model_coarse = VeryTinyNerfModel()
    model_coarse.to(device)
    model_fine = VeryTinyNerfModel()
    model_fine.to(device)
    # optimizer = torch.optim.Adam(
    #     list(model_coarse.parameters()) + list(model_fine.parameters()), lr=5e-4
    # )
    optimizer = torch.optim.SGD(
        list(model_coarse.parameters()) + list(model_fine.parameters()), lr=5e-4
    )
    # optimizer = torch.optim.Adam(model_coarse.parameters(), lr=5e-4)
    # optimizer = torch.optim.SGD(model_coarse.parameters(), lr=1e-3)

    render_kwargs_train = {
        'perturb' : 1.,
        'N_importance' : 64,
        'network_fine' : model_fine,
        'embed_fn' : positional_encoding,
        'N_samples' : 64,
        'network_coarse' : model_coarse,
        'use_viewdirs' : True,
        'embeddirs_fn' : positional_encoding,
        'white_bkgd' : False,
        'raw_noise_std' : 0.,
        'near': 0.,
        'far': 1.,
        'ndc': True,
        'lindisp': False,
    }

    # TODO: Prepare raybatch tensor if batching random rays

    for i in range(1000):
        img_idx = np.random.choice(i_train)
        img_target = images[img_idx].to(device)
        pose_target = poses[img_idx, :3, :4].to(device)
        ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
        coords = torch.stack(meshgrid_xy(torch.arange(H).to(device),
                                         torch.arange(W).to(device)), dim=-1)
        coords = coords.reshape((-1, 2))
        select_inds = np.random.choice(coords.shape[0], size=(N_rand), replace=False)
        select_inds = coords[select_inds]
        ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
        ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
        batch_rays = torch.stack([ray_origins, ray_directions], dim=0)
        target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]
        
        rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine = run_one_iter_of_nerf(
            H, W, focal, chunksize, batch_rays, render_kwargs_train
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

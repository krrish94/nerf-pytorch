import torch

from nerf_helpers import get_minibatches, get_ray_bundle
from nerf_helpers import ndc_rays, positional_encoding
from nerf_helpers import sample_pdf_2 as sample_pdf
from volume_rendering_utils import volume_render_radiance_field


def run_network(network_fn, pts, ray_batch, chunksize, embed_fn,
                embeddirs_fn):

    pts_flat = pts.reshape((-1, pts.shape[-1]))
    embedded = embed_fn(pts_flat)
    if embeddirs_fn is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)

    batches = get_minibatches(embedded, chunksize=chunksize)
    preds = []
    for batch in batches:
        preds.append(network_fn(batch))
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(list(pts.shape[:-1]) + [radiance_field.shape[-1]])
    return radiance_field


def identity_encoding(x):
    return x


def predict_and_render_radiance(
    ray_batch, model_coarse, model_fine, options, mode="train",
    encode_position_fn=None, encode_direction_fn=None
):
    # TESTED

    if encode_position_fn is None:
        encode_position_fn = identity_encoding
    if encode_direction_fn is None:
        encode_direction_fn = identity_encoding

    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    bounds = ray_batch[..., 6:8].reshape((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    # # TODO: Use actual values for "near" and "far" (instead of 0. and 1.)
    # # when not enabling "ndc".
    # t_vals = torch.linspace(0., 1., getattr(options.nerf, mode).num_coarse).to(ro)
    # if not getattr(options.nerf, mode).lindisp:
    #     z_vals = near * (1. - t_vals) + far * t_vals
    # else:
    #     z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    # z_vals = z_vals.expand([num_rays, getattr(options.nerf, mode).num_coarse])

    # if getattr(options.nerf, mode).perturb:
    #     # Get intervals between samples.
    #     mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    #     upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
    #     lower = torch.cat((z_vals[..., :1], mids), dim=-1)
    #     # Stratified samples in those intervals.
    #     t_rand = torch.rand(z_vals.shape).to(ro)
    #     z_vals = lower + (upper - lower) * t_rand
    # # pts -> (num_rays, N_samples, 3)
    # pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    # # encode_position_fn = eval(options.nerf.encode_position_fn)
    # # encode_direction_fn = eval(options.nerf.encode_direction_fn)
    # # if options.nerf.encode_position_fn == "positional_encoding":
    # #     encode_position_fn = positional_encoding
    # # if options.nerf.encode_direction_fn == "positional_encoding":
    # #     encode_direction_fn = positional_encoding

    num_coarse = getattr(options.nerf, mode).num_coarse
    far_ = far[0].item()
    near_ = near[0].item()
    z_vals = torch.linspace(near_, far_, num_coarse).to(ro)
    noise_shape = list(ro.shape[:-1]) + [num_coarse]
    z_vals = z_vals + torch.rand(noise_shape).to(ro) * (far_ - near_) / num_coarse
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    radiance_field = run_network(model_coarse,
                                 pts,
                                 ray_batch,
                                 getattr(options.nerf, mode).chunksize,
                                 encode_position_fn,
                                 encode_direction_fn,
                                )
    
    rgb_coarse, disp_coarse, acc_coarse, weights, depth_coarse = volume_render_radiance_field(
        radiance_field, z_vals, rd,
        radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
        white_background=getattr(options.nerf, mode).white_background
    )

    # TODO: Implement importance sampling, and finer network.
    rgb_fine, disp_fine, acc_fine = None, None, None
    if getattr(options.nerf, mode).num_fine > 0:
        # rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], getattr(options.nerf, mode).num_fine,
            det=(getattr(options.nerf, mode).perturb == 0.)
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
        # pts -> (N_rays, N_samples + N_importance, 3)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

        radiance_field = run_network(model_fine,
                                     pts,
                                     ray_batch,
                                     getattr(options.nerf, mode).chunksize,
                                     encode_position_fn,
                                     encode_direction_fn
                                    )
        rgb_fine, disp_fine, acc_fine, _, _ = volume_render_radiance_field(
            radiance_field, z_vals, rd,
            radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background
        )

    return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine


def run_one_iter_of_nerf(
    H, W, focal, model_coarse, model_fine, batch_rays, options, mode="train",
    encode_position_fn=None, encode_direction_fn=None
):
    if encode_position_fn is None:
        encode_position_fn = identity_encoding
    if encode_direction_fn is None:
        encode_direction_fn = identity_encoding

    ray_origins = batch_rays[0]
    ray_directions = batch_rays[1]
    viewdirs = None
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.reshape((-1, 3))
    ray_shapes = ray_directions.shape  # Cache now, to restore later.
    ro, rd = ndc_rays(H, W, focal, 1., ray_origins, ray_directions)
    ro = ro.reshape((-1, 3))
    rd = rd.reshape((-1, 3))
    near = options.nerf.near * torch.ones_like(rd[..., :1])
    far = options.nerf.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)
    if options.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
    # TODO: Init a list, keep appending outputs to that list,
    # concat everything in the end.
    rgb_coarse, disp_coarse, acc_coarse = [], [], []
    rgb_fine, disp_fine, acc_fine = None, None, None
    for batch in batches:
        rc, dc, ac, rf, df, af = predict_and_render_radiance(
            batch, model_coarse, model_fine, options,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn
        )
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

    return rgb_coarse_, disp_coarse_, acc_coarse_, rgb_fine_, disp_fine_, acc_fine_


def eval_nerf(height, width, focal_length, model_coarse, model_fine,
              ray_origins, ray_directions, options, mode="validation",
              encode_position_fn=None, encode_direction_fn=None):
    r"""Evaluate a NeRF by synthesizing a full image (as opposed to train mode, where
    only a handful of rays/pixels are synthesized).
    """
    if encode_position_fn is None:
        encode_position_fn = identity_encoding
    if encode_direction_fn is None:
        encode_direction_fn = identity_encoding
    original_shape = ray_origins.shape
    ray_origins = ray_origins.reshape((1, -1, 3))
    ray_directions = ray_directions.reshape((1, -1, 3))
    batch_rays = torch.cat((ray_origins, ray_directions), dim=0)
    rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
        height, width, focal_length, model_coarse, model_fine, batch_rays, options, mode="validation",
        encode_position_fn=encode_position_fn, encode_direction_fn=encode_direction_fn
    )
    rgb_coarse = rgb_coarse.reshape(original_shape)
    if rgb_fine is not None:
        rgb_fine = rgb_fine.reshape(original_shape)

    return rgb_coarse, None, None, rgb_fine, None, None

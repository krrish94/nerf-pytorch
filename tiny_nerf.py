import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm, trange

from nerf import cumprod_exclusive, get_minibatches, get_ray_bundle, positional_encoding


def compute_query_points_from_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near_thresh: float,
    far_thresh: float,
    num_samples: int,
    randomize: Optional[bool] = True,
) -> (torch.Tensor, torch.Tensor):
    r"""Compute query 3D points given the "bundle" of rays. The near_thresh and far_thresh
    variables indicate the bounds within which 3D points are to be sampled.

    Args:
        ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
          `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
        ray_directions (torch.Tensor): Direction of each ray in the "bundle" as returned by the
          `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
        near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
          coordinate that is of interest/relevance).
        far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
          coordinate that is of interest/relevance).
        num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
          randomly, whilst trying to ensure "some form of" uniform spacing among them.
        randomize (optional, bool): Whether or not to randomize the sampling of query points.
          By default, this is set to `True`. If disabled (by setting to `False`), we sample
          uniformly spaced points along each ray in the "bundle".

    Returns:
        query_points (torch.Tensor): Query points along each ray
          (shape: :math:`(width, height, num_samples, 3)`).
        depth_values (torch.Tensor): Sampled depth values along each ray
          (shape: :math:`(num_samples)`).
    """
    # TESTED
    # shape: (num_samples)
    depth_values = torch.linspace(near_thresh, far_thresh, num_samples).to(ray_origins)
    if randomize is True:
        # ray_origins: (width, height, 3)
        # noise_shape = (width, height, num_samples)
        noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
        # depth_values: (num_samples)
        depth_values = (
            depth_values
            + torch.rand(noise_shape).to(ray_origins)
            * (far_thresh - near_thresh)
            / num_samples
        )
    # (width, height, num_samples, 3) = (width, height, 1, 3) + (width, height, 1, 3) * (num_samples, 1)
    # query_points:  (width, height, num_samples, 3)
    query_points = (
        ray_origins[..., None, :]
        + ray_directions[..., None, :] * depth_values[..., :, None]
    )
    # TODO: Double-check that `depth_values` returned is of shape `(num_samples)`.
    return query_points, depth_values


def render_volume_density(
    radiance_field: torch.Tensor, ray_origins: torch.Tensor, depth_values: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    r"""Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
    radiance_field (torch.Tensor): A "field" where, at each query location (X, Y, Z),
      we have an emitted (RGB) color and a volume density (denoted :math:`\sigma` in
      the paper) (shape: :math:`(width, height, num_samples, 4)`).
    ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
      `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
      (shape: :math:`(num_samples)`).

    Returns:
    rgb_map (torch.Tensor): Rendered RGB image (shape: :math:`(width, height, 3)`).
    depth_map (torch.Tensor): Rendered depth image (shape: :math:`(width, height)`).
    acc_map (torch.Tensor): # TODO: Double-check (I think this is the accumulated
      transmittance map).
    """
    # TESTED
    sigma_a = torch.nn.functional.relu(radiance_field[..., 3])
    rgb = torch.sigmoid(radiance_field[..., :3])
    one_e_10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(-1)

    return rgb_map, depth_map, acc_map


# One iteration of TinyNeRF (forward pass).
def run_one_iter_of_tinynerf(
    height,
    width,
    focal_length,
    tform_cam2world,
    near_thresh,
    far_thresh,
    depth_samples_per_ray,
    encoding_function,
    get_minibatches_function,
    chunksize,
    model,
    encoding_function_args,
):

    # Get the "bundle" of rays through all image pixels.
    ray_origins, ray_directions = get_ray_bundle(
        height, width, focal_length, tform_cam2world
    )

    # Sample query points along each ray
    query_points, depth_values = compute_query_points_from_rays(
        ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray
    )

    # "Flatten" the query points.
    flattened_query_points = query_points.reshape((-1, 3))

    # Encode the query points (default: positional encoding).
    encoded_points = encoding_function(flattened_query_points, encoding_function_args)

    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    batches = get_minibatches_function(encoded_points, chunksize=chunksize)
    predictions = []
    for batch in batches:
        predictions.append(model(batch))
    radiance_field_flattened = torch.cat(predictions, dim=0)

    # "Unflatten" to obtain the radiance field.
    unflattened_shape = list(query_points.shape[:-1]) + [4]
    radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_predicted, _, _ = render_volume_density(
        radiance_field, ray_origins, depth_values
    )

    return rgb_predicted


class VeryTinyNerfModel(torch.nn.Module):
    r"""Define a "very tiny" NeRF model comprising three fully connected layers.
    """

    def __init__(self, filter_size=128, num_encoding_functions=6):
        super(VeryTinyNerfModel, self).__init__()
        # Input layer (default: 39 -> 128)
        self.layer1 = torch.nn.Linear(3 + 3 * 2 * num_encoding_functions, filter_size)
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


def main():

    # Determine device to run on (GPU vs CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log directory
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "log")
    os.makedirs(logdir, exist_ok=True)

    """
    Load input images and poses
    """

    data = np.load("cache/tiny_nerf_data.npz")

    # Images
    images = data["images"]
    # Camera extrinsics (poses)
    tform_cam2world = data["poses"]
    tform_cam2world = torch.from_numpy(tform_cam2world).to(device)
    # Focal length (intrinsics)
    focal_length = data["focal"]
    focal_length = torch.from_numpy(focal_length).to(device)

    # Height and width of each image
    height, width = images.shape[1:3]

    # Near and far clipping thresholds for depth values.
    near_thresh = 2.0
    far_thresh = 6.0

    # Hold one image out (for test).
    testimg, testpose = images[101], tform_cam2world[101]
    testimg = torch.from_numpy(testimg).to(device)

    # Map images to device
    images = torch.from_numpy(images[:100, ..., :3]).to(device)

    """
    Parameters for TinyNeRF training
    """

    # Number of functions used in the positional encoding (Be sure to update the
    # model if this number changes).
    num_encoding_functions = 10
    # Specify encoding function.
    encode = positional_encoding
    # Number of depth samples along each ray.
    depth_samples_per_ray = 32

    # Chunksize (Note: this isn't batchsize in the conventional sense. This only
    # specifies the number of rays to be queried in one go. Backprop still happens
    # only after all rays from the current "bundle" are queried and rendered).
    # Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory (when using 8
    # samples per ray).
    chunksize = 4096

    # Optimizer parameters
    lr = 5e-3
    num_iters = 5000

    # Misc parameters
    display_every = 100  # Number of iters after which stats are

    """
    Model
    """
    model = VeryTinyNerfModel(num_encoding_functions=num_encoding_functions)
    model.to(device)

    """
    Optimizer
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    """
    Train-Eval-Repeat!
    """

    # Seed RNG, for repeatability
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []

    for i in trange(num_iters):

        # Randomly pick an image as the target.
        target_img_idx = np.random.randint(images.shape[0])
        target_img = images[target_img_idx].to(device)
        target_tform_cam2world = tform_cam2world[target_img_idx].to(device)

        # Run one iteration of TinyNeRF and get the rendered RGB image.
        rgb_predicted = run_one_iter_of_tinynerf(
            height,
            width,
            focal_length,
            target_tform_cam2world,
            near_thresh,
            far_thresh,
            depth_samples_per_ray,
            encode,
            get_minibatches,
            chunksize,
            model,
            num_encoding_functions,
        )

        # Compute mean-squared error between the predicted and target images. Backprop!
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Display images/plots/stats
        if i % display_every == 0 or i == num_iters - 1:
            # Render the held-out view
            rgb_predicted = run_one_iter_of_tinynerf(
                height,
                width,
                focal_length,
                testpose,
                near_thresh,
                far_thresh,
                depth_samples_per_ray,
                encode,
                get_minibatches,
                chunksize,
                model,
                num_encoding_functions,
            )
            loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
            tqdm.write("Loss: " + str(loss.item()))
            psnr = -10.0 * torch.log10(loss)

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.imshow(rgb_predicted.detach().cpu().numpy())
            plt.savefig(os.path.join(logdir, str(i).zfill(6) + ".png"))
            plt.close("all")

            if i == num_iters - 1:
                plt.plot(iternums, psnrs)
                plt.savefig(os.path.join(logdir, "psnr.png"))
                plt.close("all")
            # plt.figure(figsize=(10, 4))
            # plt.subplot(121)
            # plt.imshow(rgb_predicted.detach().cpu().numpy())
            # plt.title(f"Iteration {i}")
            # plt.subplot(122)
            # plt.plot(iternums, psnrs)
            # plt.title("PSNR")
            # plt.show()

    print("Done!")


if __name__ == "__main__":
    # m = TinyNerfModel(depth=8)
    # m.cuda()
    # print(m)
    # print(m(torch.rand(2, 39).cuda()).shape)
    main()

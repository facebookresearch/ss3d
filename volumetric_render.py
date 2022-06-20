import json
import math
import os
import random
import sys
import time

import imageio
import numpy as np
import pytorch3d
import pytorch3d.renderer
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange


def network_query_fn_train(inputs, c, decoder, use_abs=False):
    # inputs shape : [N_rays, N_samples, 3]
    # c shpae: either [N_rays,c_size]
    if use_abs:
        return torch.abs(decoder(inputs, c=c))

    return decoder(inputs, c=c)


def network_query_fn_validation(inputs, c, decoder, use_relu=False):
    # inputs shape : [ray_chunk, N_samples, 3]
    # c shpae: either [1,c_size]
    assert c.shape[0] == 1, "C should be of shape 1*c_dim for val"

    c_inp = torch.cat(inputs.shape[0] * [c])
    out = decoder(inputs, c=c_inp.type_as(inputs))

    flags_x = (inputs[..., 0] > 0.5) * (inputs[..., 0] < -0.5)
    flags_y = (inputs[..., 1] > 0.5) * (inputs[..., 1] < -0.5)
    flags_z = (inputs[..., 2] > 0.5) * (inputs[..., 2] < -0.5)

    flags = flags_x * flags_y * flags_z
    flags = torch.stack([flags, flags, flags, flags], dim=-1)

    outs = decoder(inputs, c=c)
    outs[flags] = 0.0
    return outs


def get_transformation(dist, elev, azim):
    # returns camera to world
    R, T = pytorch3d.renderer.look_at_view_transform(dist, elev, azim)
    T_rays = pytorch3d.renderer.camera_position_from_spherical_angles(dist, elev, azim)
    c2w = torch.cat((R, T_rays.reshape(1, 3, 1)), -1)[0]
    return torch.cat([c2w, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)


def get_rays_from_angles(
    *,
    H,
    W,
    focal,
    near_plane,
    far_plane,
    elev_angles,
    azim_angles,
    dists,
    device,
    indices,
    transformation_rel=None,  # 4x4 homogeneous matrix encoder to world
):
    rays = []
    for i in range(elev_angles.shape[0]):
        if transformation_rel is not None:
            transformation_view_to_world = get_transformation(
                dists[i], elev_angles[i], azim_angles[i]
            )
            # view to encoder
            c2w = torch.matmul(
                transformation_rel.inverse(), transformation_view_to_world
            )
            c2w = c2w[:3, :]
        else:
            R, T = pytorch3d.renderer.look_at_view_transform(
                dists[i], elev_angles[i], azim_angles[i]
            )
            T_rays = pytorch3d.renderer.camera_position_from_spherical_angles(
                dists[i], elev_angles[i], azim_angles[i]
            )
            c2w = torch.cat((R, T_rays.reshape(1, 3, 1)), -1)[0]
        rays_o, rays_d = get_rays(
            H, W, focal, c2w
        )  # (rays_o = [H, W,3], rays_d = [H, W,3])
        rays_o = rays_o.reshape(-1, 3)  # [H*W, 3]
        rays_d = rays_d.reshape(-1, 3)  # [H*W, 3]
        rays_o = rays_o[indices[i]].to(device)  # [num_rays, 3]
        rays_d = rays_d[indices[i]].to(device)  # [num_rays, 3]

        near, far = (
            # near_plane * torch.ones_like(rays_d[..., :1]),
            # far_plane * torch.ones_like(rays_d[..., :1]),
            max(dists[i].to(device) - 0.90, 0.0) * torch.ones_like(rays_d[..., :1]),
            (dists[i].to(device) + 0.90) * torch.ones_like(rays_d[..., :1]),
        )  # [num_rays, 1], # [num_rays, 1]
        rays.append(
            torch.cat([rays_o, rays_d, near, far], -1).to(device)
        )  # List([num_rays, 8])

    return torch.cat(rays, dim=0)  # [N*num_rays, 8] (N=Batch size)


# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H)
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack(
        [-(i - W * 0.5) / focal, -(j - H * 0.5) / focal, torch.ones_like(i)], -1
    )  # https://pytorch3d.org/docs/renderer_getting_started
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d  # (rays_o = [H, W,3], rays_d = [H, W,3])


def raw2outputs(
    raw,
    z_vals,
    rays_d,
    raw_noise_std=0,
    white_bkgd=False,
    has_rgb=True,
    render_depth=False,
    raw_normals=None,
):
    if white_bkgd:
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(
            -act_fn(5 * raw) * dists
        )
    else:
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(
            -act_fn(raw) * dists
        )

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).type_as(z_vals)], -1
    )  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    if has_rgb:
        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    if has_rgb:
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples] # Ex: Nert
    else:
        alpha = raw2alpha(raw + noise, dists)  # [N_rays, N_samples] # Ex: Onet

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = (
        alpha  # n(sample) W_n = alpha_n * (prod_{j<n}(1-alpha_j))
        * torch.cumprod(
            torch.cat(
                [torch.ones((alpha.shape[0], 1)).type_as(z_vals), 1.0 - alpha + 1e-10],
                -1,
            ),
            -1,
        )[:, :-1]
    )  # [N_rays, N_samples]

    ret = {}
    if render_depth:
        # Ignore the last point weight. I.e the point at infinity
        depth_map = torch.sum(weights[:, :-1] * z_vals[:, :-1], -1)
        ret["depth_map"] = depth_map
        disp_map = 1.0 / torch.max(
            1e-10 * torch.ones_like(depth_map),
            depth_map / torch.sum(weights[:, :-1], -1),
        )
        ret["disp_map"] = disp_map

    # Ignore the last point weight. I.e the point at infinity
    acc_map = torch.sum(weights[:, :-1], -1)
    # acc_map = torch.sum(weights, -1)
    ret["acc_map"] = acc_map

    if has_rgb:
        # rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        rgb_map = torch.sum(weights[:, :-1, None] * rgb[:, :-1], -2)  # [N_rays, 3]
        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])
        ret["rgb_map"] = rgb_map

    if raw_normals is not None:
        normal_map = torch.sum(
            weights[:, :-1, None] * raw_normals[:, :-1], -2
        )  # [N_rays, 3]
        ret["normal_map"] = normal_map

    ret["weights"] = weights
    return ret


def render_rays(
    ray_batch,
    c_latent,
    decoder,
    N_samples,
    network_query_fn=network_query_fn_train,
    retraw=False,
    white_bkgd=False,
    raw_noise_std=0.0,
    has_rgb=False,
    has_normal=False,
    val_mode=False,
):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    t_vals = torch.linspace(0.0, 1.0, steps=N_samples).type_as(ray_batch)
    z_vals = near * (1.0 - t_vals) + far * (t_vals)
    z_vals = z_vals.expand([N_rays, N_samples])

    pts = (
        rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    )  # [N_rays, N_samples, 3]

    if has_normal:
        if val_mode:
            torch.set_grad_enabled(True)
            decoder.train()
        pts.requires_grad = True

    raw = network_query_fn(pts, c_latent, decoder)  # [N_rays, N_samples]

    # Estimate Normals
    raw_normals = None
    if has_normal:
        temp_sum = F.relu(raw[..., 3]).sum() if has_rgb else F.relu(raw).sum()
        raw_normals = torch.autograd.grad(
            outputs=temp_sum, inputs=pts, create_graph=True, retain_graph=True
        )
        raw_normals = raw_normals[0]
        # raw_normals = raw_normals / torch.norm(raw_normals, dim=-1, keepdim=True)
        raw_normals = -1 * torch.nn.functional.normalize(raw_normals, eps=1e-6, dim=-1)
        # [-1,1] to [0,1]
        raw_normals = raw_normals * 0.5 + 0.5
        if val_mode:
            torch.set_grad_enabled(False)
            decoder.eval()

    ret = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, has_rgb, raw_normals=raw_normals
    )  # Dict

    if retraw:
        ret["raw"] = raw
        ret["points"] = pts

    # for k in ret:
    #     if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
    #         print(f"! [Numerical Error] {k} contains nan or inf.", k)

    return ret


def batchify_rays(rays_flat, chunk, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    c_latent = kwargs["c_latent"]
    for i in range(0, rays_flat.shape[0], chunk):
        kwargs["c_latent"] = c_latent  # [i:i+chunk]
        ret = render_rays(rays_flat[i : i + chunk], **kwargs)

        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

    return all_ret


def render(
    H,
    W,
    focal,
    device,
    chunk=1024,
    rays=None,
    c2w=None,
    near=0.0,
    far=2.0,
    render_dist=None,
    use_relative=False,
    **kwargs,
):
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    # TODO: Check this!!!!!
    if render_dist is not None:
        limits = 0.9
        if use_relative:
            limits = 1.10
        near, far = (
            # near_plane * torch.ones_like(rays_d[..., :1]),
            # far_plane * torch.ones_like(rays_d[..., :1]),
            max(render_dist.to("cpu") - limits, 0.0) * torch.ones_like(rays_d[..., :1]),
            (render_dist.to("cpu") + limits) * torch.ones_like(rays_d[..., :1]),
        )  # [num_rays, 1], # [num_rays, 1]
    else:
        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(
            rays_d[..., :1]
        )
    rays = torch.cat([rays_o, rays_d, near, far], -1).to(device)

    # Render and  reshape to match img dimentions
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    return all_ret


def render_img(
    dist,
    elev_angle,
    azim_angle,
    img_size,
    focal,
    render_kwargs,
    render_factor=0,
):

    H, W = img_size, img_size
    R, T = pytorch3d.renderer.look_at_view_transform(dist, elev_angle, azim_angle)
    T_rays = pytorch3d.renderer.camera_position_from_spherical_angles(
        dist, elev_angle, azim_angle
    )
    c2w_mat = torch.cat((R, T_rays.reshape(1, 3, 1)), -1)[0]

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    all_ret = render(
        H,
        W,
        focal,
        c2w=c2w_mat[:3, :4],
        render_dist=dist,
        use_relative=False,
        **render_kwargs,
    )
    depth_img = all_ret.get("depth_map", None)
    occ_img = all_ret["acc_map"]
    rgb_img = all_ret.get("rgb_map", None)
    normal_img = all_ret.get("normal_map", None)

    return depth_img, occ_img, rgb_img, normal_img

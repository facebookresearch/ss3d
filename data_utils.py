# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import imageio
import mcubes
import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as F
import trimesh
from PIL import Image
from torchvision import transforms


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")


def get_tight_bbox(mask):
    mask_bool = mask > 0.3
    row_agg = mask_bool.sum(dim=0)
    row_agg = row_agg > 0
    col_agg = mask_bool.sum(dim=1)
    col_agg = col_agg > 0

    def get_left_right(x):
        left = 0
        for i in range(len(x)):
            if x[i]:
                left = i
                break
        right = len(x)
        for i in range(len(x) - 1, 0, -1):
            if x[i]:
                right = i
                right += 1
                break

        return left, right

    x1, x2 = get_left_right(col_agg)
    y1, y2 = get_left_right(row_agg)

    return x1, x2, y1, y2


def generate_input_img(
    rgb_path,
    mask_path,
):
    inp_transforms = transforms.Compose(
        [
            SquarePad(),  # pad to square
            transforms.Pad(30, fill=0, padding_mode="constant"),
            # functional.crop,
            transforms.Resize((224, 224)),  # resize
            # transforms.Normalize(3 * [0.5], 3 * [0.5]),
            transforms.ToTensor(),
        ]
    )

    with open(rgb_path, "rb") as f:
        raw_rgb_img = Image.open(f)
        raw_rgb_img = np.array(raw_rgb_img.convert("RGB"))

    mask_image = imageio.imread(mask_path)
    mask_image = (torch.Tensor(mask_image)).float() / 255.0

    # clip based on bbox
    bbox = get_tight_bbox(mask_image)
    label_img = mask_image[bbox[0] : bbox[1], bbox[2] : bbox[3]]

    rgb_img = (
        raw_rgb_img[bbox[0] : bbox[1], bbox[2] : bbox[3], :]
        * label_img.unsqueeze(-1).numpy()
    )

    return (
        inp_transforms(PIL.Image.fromarray(rgb_img.astype(np.uint8)))
        .float()
        .unsqueeze(0)
    )


def extract_trimesh(model, img, device="cuda", threshold=3.0, discretization=100):

    model = model.to(device)

    c_latent = model.encoder(img.to(device))
    assert c_latent.shape[0] == 1, "C should be of shape 1*c_dim for val"

    # Volume during training is contained ot a cube of [-0.5,0.5]
    x_l = torch.FloatTensor(np.linspace(-0.5, 0.5, discretization)).to(device)
    y_l = torch.FloatTensor(np.linspace(-0.5, 0.5, discretization)).to(device)
    z_l = torch.FloatTensor(np.linspace(-0.5, 0.5, discretization)).to(device)
    x, y, z = torch.meshgrid(x_l, y_l, z_l)

    points_cords = torch.stack([x, y, z], dim=-1)
    with torch.no_grad():

        # c_inp = torch.cat(points_cords.shape[0] * [c])
        # pred_voxels = network_query_fn_validation(points_cords, c_latent, model.decoder)

        pred_voxels = model.decoder(points_cords, c=c_latent)
        pred_voxels = pred_voxels[..., 3]
        pred_voxels = pred_voxels.cpu().numpy()

    vertices, triangles = mcubes.marching_cubes(pred_voxels, threshold)
    return trimesh.Trimesh(vertices, triangles, vertex_normals=None, process=False)

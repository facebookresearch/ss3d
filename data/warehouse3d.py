import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import os.path as osp
import random
import time

import imageio
import numpy as np
import scipy.io as sio
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DistributedSampler
from volumetric_render import (
    get_rays_from_angles,
    get_transformation,
)


def loadDepth(dFile, minVal=0, maxVal=10):
    dMap = imageio.imread(dFile)
    dMap = dMap.astype(np.float32)
    dMap = dMap * (maxVal - minVal) / (pow(2, 16) - 1) + minVal
    return dMap


def getSynsetsV1Paths(data_cfg):
    synsets = data_cfg.class_ids.split(",")
    synsets.sort()
    root_dir = data_cfg.rgb_dir
    synsetModels = [
        [f for f in os.listdir(osp.join(root_dir, s)) if len(f) > 3] for s in synsets
    ]

    paths = []
    for i in range(len(synsets)):
        for m in synsetModels[i]:
            paths.append([synsets[i], m])

    return paths


def get_rays_multiplex(cameras, rgb_imgs, mask_imgs, render_cfg, device):
    len_cameras = len(cameras)
    assert len_cameras == len(rgb_imgs) and len_cameras == len(
        mask_imgs
    ), "incorrect inputs for camera computation"

    all_rays = []
    all_rgb_labels = []
    all_mask_labels = []
    for i, cam in enumerate(cameras):
        # compute rays

        assert len(cam) == render_cfg.cam_num, "incorrect per frame camera number"
        indices = []
        ind = torch.randperm(render_cfg.img_size * render_cfg.img_size)
        for j in range(len(cam)):
            indices.append(ind[: render_cfg.ray_num_per_cam])
            all_rgb_labels.append(rgb_imgs[i, indices[-1]])
            all_mask_labels.append(mask_imgs[i, indices[-1]])

        all_rays.append(
            get_rays_from_angles(
                H=render_cfg.img_size,
                W=render_cfg.img_size,
                focal=float(render_cfg.focal_length),
                near_plane=render_cfg.near_plane,
                far_plane=render_cfg.far_plane,
                elev_angles=cam[:, 0],
                azim_angles=cam[:, 1],
                dists=cam[:, 2],
                device=device,
                indices=indices,
            )
        )  # [(Num_cams_per_frame*Num_rays), 8] #2d

    return (
        torch.cat(all_mask_labels).to(
            device
        ),  # [(N*Num_cams_per_frame*Num_rays_per_cam)] #1d
        torch.cat(all_rgb_labels).to(
            device
        ),  # [[(N*Num_cams_per_frame*Num_rays_per_cam), 3] #2d
        torch.cat(all_rays).to(
            device
        ),  # [(N*Num_cams_per_frame*Num_rays_per_cam), 8] #2d
    )


def extract_data_train(batch_dict, render_cfg, device):
    # If using pre-rendered.
    assert "mask_label_rays" in batch_dict

    inp_imgs = batch_dict["rgb_img"]
    mask_label_rays = batch_dict["mask_label_rays"].view(-1)
    rays = batch_dict["rays"].view(-1, 8)
    rgb_label_rays = batch_dict["rgb_label_rays"]
    rgb_label_rays = rgb_label_rays.reshape(-1, 3)

    return (
        inp_imgs.to(device),  # [N, 3, img_size, img_size]
        mask_label_rays.to(device),  # [(N*Num_rays)] #1d
        rgb_label_rays.to(device),  # [[(N*Num_rays), 3] #2d
        None,
        None,
        rays.to(device),  # [(N*Num_rays), 8] #2d
    )


class DatasetPermutationWrapper(Dataset):
    def __init__(self, dset):
        self.dset = dset
        self._len = len(self.dset)

    def __len__(self):
        return self._len

    def __getitem__(self, _):
        # TODO(Fix): This random generator behaves same on all gpu's
        index = random.randint(0, self._len - 1)
        return self.dset[index]


class WareHouse3DDataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    # TODO: Change hardcoded values!
    def __init__(self, data_root, paths, render_cfg, encoder_root):

        super(WareHouse3DDataset, self).__init__()

        self.paths = paths
        self.render_cfg = render_cfg
        self.data_root = data_root
        self.n_cams = self.render_cfg.cam_num
        self.n_rays_per_cam = self.render_cfg.ray_num_per_cam
        self.encoder_root = encoder_root

        self.transform_img = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        self.transform_label = transforms.Compose(
            [
                transforms.Resize((self.render_cfg.img_size, self.render_cfg.img_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        st_time = time.time()
        rel_path = os.path.join(self.paths[index][0], self.paths[index][1])
        data_path = os.path.join(self.data_root, rel_path)

        # sample a random encoder imput rgb image.

        encoder_cam_info = np.load(
            osp.join(self.encoder_root, rel_path, "cam_info.npy")
        )
        encoder_sample_num = random.randint(0, encoder_cam_info.shape[0] - 1)
        # encoder_sample_num = 4 # Only for debug
        inp_angles = encoder_cam_info[encoder_sample_num, :]
        inp_angles[0] += 90
        img_path = os.path.join(
            self.encoder_root, rel_path, "render_{}.png".format(encoder_sample_num)
        )
        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        inp_rgb_size = img.size[0]
        inp_focal = (sio.loadmat(os.path.join(data_path, "camera_0.mat")))["K"][0, 0]

        img = self.transform_img(img)

        # Sample random cameras
        cam_info = np.load(osp.join(data_path, "cam_info.npy"))

        # Only use a subset of the data
        if self.render_cfg.num_pre_rend_masks > 1:
            cam_info = cam_info[: self.render_cfg.num_pre_rend_masks, :]

        # TODO(Fix): This random generator behaves same on all data-workers on each gpu
        cam_inds = np.random.choice(cam_info.shape[0], self.n_cams)
        # cam_inds = [0] # only for debug
        cam_info = torch.Tensor(cam_info[cam_inds, :])
        azim_angle, elev_angle, theta, dist = torch.split(cam_info, 1, dim=-1)
        azim_angle += (
            90  # This a known blender offset. Look at the noteboosks for visual test.
        )
        # sample rays from cameras and mask images
        render_cfg = self.render_cfg
        pixel_ids = []
        ray_mask_labels = []
        ray_rgb_labels = []

        for i, nc in enumerate(cam_inds):
            temp_idx = torch.randperm(
                self.render_cfg.img_size * self.render_cfg.img_size
            )
            idx = temp_idx[: self.n_rays_per_cam]  # [1, n_rays_per_cam]
            pixel_ids.append(idx)

            # Masks from depth
            gt_mask = loadDepth(
                osp.join(data_path, "depth_{}.png".format(int(nc))), minVal=0, maxVal=10
            )
            empty = gt_mask >= 10.0
            notempty = gt_mask < 10.0
            gt_mask[empty] = 0
            gt_mask[notempty] = 1.0
            gt_mask = self.transform_label(Image.fromarray(gt_mask))
            gt_mask = gt_mask.view(-1).float()
            ray_mask_labels.append(gt_mask[idx])

            # RGB Pixels
            label_rgb_path = osp.join(data_path, "render_{}.png".format(int(nc)))
            with open(label_rgb_path, "rb") as f:
                gt_rgb = Image.open(f)
                gt_rgb = gt_rgb.convert("RGB")

            gt_rgb = self.transform_label(gt_rgb)
            gt_rgb = gt_rgb.permute(1, 2, 0)
            gt_rgb = gt_rgb.reshape(-1, 3)
            ray_rgb_labels.append(gt_rgb[idx])

            # inp_angles = cam_info[i, :]  # TODO: Test!!!

        # n_cams X n_rays_per_cam
        pixel_idx = torch.stack(pixel_ids, dim=0)
        mask_label_rays = torch.cat(ray_mask_labels, dim=0)
        rgb_label_rays = torch.cat(ray_rgb_labels, dim=0)

        label_focal = inp_focal * float(render_cfg.img_size) / inp_rgb_size

        # Used only in relative case
        rays = get_rays_from_angles(
            H=render_cfg.img_size,
            W=render_cfg.img_size,
            focal=label_focal,
            near_plane=render_cfg.near_plane,
            far_plane=render_cfg.far_plane,
            elev_angles=elev_angle[:, 0],
            azim_angles=azim_angle[:, 0],
            dists=dist[:, 0],
            device=torch.device("cpu"),
            indices=pixel_idx,
            transformation_rel=None,
        )  # [(n_cams * n_rays_per_cam), 8] #2d

        return {
            "rgb_img": img,
            "rays": rays,
            "mask_label_rays": mask_label_rays,
            "rgb_label_rays": rgb_label_rays,
            # info useful for debugging
            "elev_angle": torch.tensor([elev_angle[-1, 0]]).float(),
            "azim_angle": torch.tensor([azim_angle[-1, 0]]).float(),
            "dist": torch.tensor([dist[-1, 0]]).float(),
            "rel_path": rel_path,
            # Used for no camera pose
            "label_img_path": label_rgb_path,
            "label_rgb_img": gt_rgb,
            "label_mask_img": gt_mask,
            # class id's
            "class_id": self.paths[index][0],
            "orig_img_path": label_rgb_path,
        }


class WareHouse3DModule(LightningDataModule):
    def __init__(self, data_cfg, render_cfg, num_workers=0, debug_mode=False):
        super().__init__()
        self.data_cfg = data_cfg
        self.render_cfg = render_cfg
        self.num_workers = num_workers

        paths = getSynsetsV1Paths(self.data_cfg)

        train_size = int(self.data_cfg.train_split * len(paths))
        validation_size = int(self.data_cfg.validation_split * len(paths))
        test_size = len(paths) - train_size - validation_size
        (
            self.train_split,
            self.validation_split,
            self.test_split,
        ) = torch.utils.data.random_split(
            paths, [train_size, validation_size, test_size]
        )
        print(
            "Total Number of paths:",
            len(paths),
            len(self.train_split),
            len(self.validation_split),
        )

    def train_dataloader(self):

        assert self.render_cfg.cam_num > 0, "camera number cannot be 0"
        train_ds = WareHouse3DDataset(
            self.data_cfg.rgb_dir,
            self.train_split,
            self.render_cfg,
            self.data_cfg.encoder_dir,
        )
        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(train_ds)

        return torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.data_cfg.bs_train,
            num_workers=self.num_workers,
            shuffle=sampler is None,
            sampler=sampler,
        )

    def val_dataloader(self):

        assert self.render_cfg.cam_num > 0, "camera number cannot be 0"

        val_ds = DatasetPermutationWrapper(
            WareHouse3DDataset(
                self.data_cfg.rgb_dir,
                self.validation_split,
                self.render_cfg,
                self.data_cfg.encoder_dir,
            )
        )
        return torch.utils.data.DataLoader(
            val_ds, batch_size=self.data_cfg.bs_val, num_workers=self.num_workers
        )

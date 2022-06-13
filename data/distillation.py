import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import os
import os.path as osp
import random

import imageio
import numpy as np
import PIL
import pytorch3d
import pytorch3d.renderer
import scipy
import scipy.io
import scipy.misc
import torch
import torch.nn.functional as torch_F
import torchvision.transforms.functional as F
import trimesh
from data.generic_img_mask_loader import GenericImgMaskModule
from data.warehouse3d import WareHouse3DModule
from hydra_config import config
from model import get_model
from PIL import Image
from pytorch_lightning import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import Dataset, DistributedSampler
from torchvision import transforms


class VolumetricNetworkCkpt(LightningModule):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg

        # make the e2e (encoder + decoder) model.
        self.model = get_model(cfg.model)

        # Save hyperparameters
        self.save_hyperparameters(cfg)

        cam_num = self.cfg.render.cam_num if self.cfg.render.cam_num > 0 else 1
        self.ray_num = cam_num * self.cfg.render.ray_num_per_cam


def get_pl_datamodule(cfg):
    if cfg.data.name == "generic_img_mask":
        dataLoaderMethod = GenericImgMaskModule
    else:
        dataLoaderMethod = WareHouse3DModule

    # configure data loader
    data_loader = dataLoaderMethod(
        data_cfg=cfg.data,
        render_cfg=cfg.render,
        num_workers=cfg.resources.num_workers,
    )
    return data_loader


class DistillationDataset(Dataset):
    def __init__(self, dataset_list, cfg):

        self.index_to_dataset = []
        self.warehouse3d_probability = cfg.distillation.warehouse3d_prob

        if self.warehouse3d_probability > 0.0:
            self.dataset_list = dataset_list[1:]
            self.warehouse3d_ds = dataset_list[0]
        else:
            self.dataset_list = dataset_list
            self.warehouse3d_ds = None

        for dataset_id, dataset in enumerate(self.dataset_list):
            length = len(dataset)
            dataset_list = [(dataset_id, i) for i in range(len(dataset))]
            self.index_to_dataset += dataset_list

        render_cfg = cfg.render

        self.dist_range = [
            render_cfg.camera_near_dist,
            render_cfg.camera_far_dist,
        ]
        self.ray_num = render_cfg.ray_num_per_cam * (
            render_cfg.cam_num if render_cfg.cam_num > 0 else 1
        )
        self.img_size = render_cfg.img_size

    def __len__(self):
        return len(self.index_to_dataset)

    def __getitem__(self, index):

        if self.warehouse3d_probability == 0.0:
            dataset_id, data_idx = self.index_to_dataset[index]
            data_dict = self.dataset_list[dataset_id][data_idx]
        else:
            if torch.rand(1).item() < self.warehouse3d_probability:
                dataset_id = 0
                data_idx = torch.randint(0, len(self.warehouse3d_ds), (1,)).item()
                data_dict = self.warehouse3d_ds[data_idx]
                data_dict["orig_img_path"] = data_dict["label_img_path"]
            else:
                dataset_id, data_idx = self.index_to_dataset[index]
                data_dict = self.dataset_list[dataset_id][data_idx]
                dataset_id += 1

        # Sample random pose
        elev_angle = 90 * (2 * torch.rand(1) - 1.0)  # [-90,90]
        azim_angle = 180 * (2 * torch.rand(1) - 1.0)  # [-180,180]
        dist = self.dist_range[0] + (
            self.dist_range[1] - self.dist_range[0]
        ) * torch.rand(1)

        temp_idx = torch.randperm(self.img_size * self.img_size)
        idx = temp_idx[: self.ray_num]  # [1, num_points]

        return {
            "rgb_img": data_dict["rgb_img"],
            "dataset_id": dataset_id,
            "data_idx": data_idx,
            "label_img_path": data_dict["label_img_path"],
            "label_rgb_img": data_dict["label_rgb_img"],
            "label_mask_img": data_dict["label_mask_img"],
            "orig_img_path": data_dict["orig_img_path"],
            "dist": dist,
            "elev_angle": elev_angle,
            "azim_angle": azim_angle,
            "flat_indices": idx,
        }


def get_paths(root_dir, epoch_num=49, exclude_names=""):
    if exclude_names != "":
        exclude_names = exclude_names.split(",")
    else:
        exclude_names = []

    out_paths = []
    for root, d_names, f_names in os.walk(root_dir):
        for f in f_names:
            if str(epoch_num) + ".ckpt" in f:

                path = os.path.join(root, f)

                is_exclude = False
                for ex in exclude_names:
                    if ex in path:
                        is_exclude = True

                if not is_exclude:
                    out_paths.append(path)

    return out_paths


class DistillationDataModule(LightningDataModule):
    def __init__(self, cfg, base_path):
        super().__init__()
        self.cfg = cfg
        self.data_module_list = []
        self.num_workers = cfg.resources.num_workers
        assert cfg.distillation.ckpts_root_dir is not None
        checkpoint_paths = get_paths(
            cfg.distillation.ckpts_root_dir,
            cfg.distillation.ckpts_epoch_num,
            exclude_names=cfg.distillation.exclude_names,
        )
        checkpoint_paths = [cfg.distillation.warehouse3d_ckpt_path] + checkpoint_paths
        print("!!!!!!!!!!!!!Loading checkpoints from root dir for dataloader!!!!!!!!")
        print(checkpoint_paths, sep="\n")
        for checkpoint_path in checkpoint_paths:
            temp_model = VolumetricNetworkCkpt.load_from_checkpoint(checkpoint_path)
            self.data_module_list.append(get_pl_datamodule(temp_model.cfg))
            del temp_model

    def train_dataloader(self):

        datasets = [dm.train_dataloader().dataset for dm in self.data_module_list]
        distillation_ds = DistillationDataset(datasets, self.cfg)
        sampler = DistributedSampler(distillation_ds)

        return torch.utils.data.DataLoader(
            distillation_ds,
            batch_size=self.cfg.data.bs_train,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=sampler is None,
        )

    def val_dataloader(self):

        return self.train_dataloader()
        # datasets = [dm.val_dataloader().dataset for dm in self.data_module_list]
        # distillation_ds = DistillationDataset(datasets, self.cfg)
        # sampler = DistributedSampler(distillation_ds)
        # #sampler = None
        # return torch.utils.data.DataLoader(
        #     distillation_ds,
        #     batch_size=self.cfg.data.bs_val,
        #     num_workers=self.num_workers,
        #     sampler=sampler,
        #     shuffle=sampler is None,
        # )

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import os.path as osp
import random

import imageio
import numpy as np
import pandas
import PIL
import torch
import torch.nn.functional as torch_F
import torchvision.transforms.functional as F
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DistributedSampler
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

    return np.array([x1, y1, x2, y2])


def read_paths_and_boxes(file_path, data_cfg):

    class_names = data_cfg.class_ids.split(",")

    split_df = pandas.read_csv(
        file_path,
        header=None,
        names=[
            "class_id",
            "rgb_path",
            "mask_path",
            "BoxXMin",
            "BoxYMin",
            "BoxXMax",
            "BoxYMax",
        ],
    )
    img_list = []
    for allowed_id in class_names:
        class_df = split_df.loc[split_df["class_id"] == allowed_id]

        # valid_df = class_df.loc[class_df["truncated"] > iou_th]
        num_images = min(len(class_df), data_cfg.max_per_class)
        assert num_images > 10, f"Minimum image criterion not met for {class_names}"
        class_df = class_df.head(num_images)

        cls_counter = 0
        for index, row in class_df.iterrows():

            rgb_path = None
            mask_path = None
            if osp.exists(
                osp.join(data_cfg.rgb_path_prefix, row["rgb_path"])
            ) and osp.exists(osp.join(data_cfg.mask_path_prefix, row["rgb_path"])):
                rgb_path = osp.join(data_cfg.rgb_path_prefix, row["rgb_path"])
                mask_path = osp.join(data_cfg.mask_path_prefix, row["rgb_path"])

            if (mask_path is None) or (rgb_path is None):
                continue

            if math.isnan(row["BoxYMax"]):
                bbox = None
            else:
                bbox = [row["BoxXMin"], row["BoxYMin"], row["BoxXMax"], row["BoxYMax"]]

            img_list.append(
                {
                    "bbox": bbox,
                    "rgb_path": rgb_path,
                    "mask_path": mask_path,
                }
            )
            cls_counter += 1

            if cls_counter > data_cfg.max_per_class:
                break
    return img_list


class GenericImgMaskDataset(Dataset):
    def __init__(self, img_list, data_cfg, render_cfg):

        self.data_cfg = data_cfg

        self.render_cfg = render_cfg
        self.img_list = img_list

        self.inp_transforms = transforms.Compose(
            [
                SquarePad(),  # pad to square
                transforms.Pad(30, fill=0, padding_mode="constant"),
                # functional.crop,
                transforms.Resize((224, 224)),  # resize
                transforms.ToTensor(),
            ]
        )

        self.label_transforms = transforms.Compose(
            [
                SquarePad(),  # pad to square
                transforms.Pad(30, fill=0, padding_mode="constant"),
                # functional.crop,
                transforms.Resize(
                    (self.render_cfg.img_size, self.render_cfg.img_size)
                ),  # resize
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        img_dict = self.img_list[index]
        img_path = img_dict["rgb_path"]
        mask_path = img_dict["mask_path"]

        with open(img_path, "rb") as f:
            raw_rgb_img = Image.open(f)
            raw_rgb_img = np.array(raw_rgb_img.convert("RGB"))

        mask_image = imageio.imread(mask_path)
        mask_image = (torch.Tensor(mask_image)).float() / 255.0

        if len(mask_image.shape) == 3:
            mask_image = mask_image[..., -1]

        mask_image = torch_F.interpolate(
            mask_image.unsqueeze(0).unsqueeze(0),
            (raw_rgb_img.shape[0], raw_rgb_img.shape[1]),
        )
        mask_image = mask_image.squeeze(0).squeeze(0)

        # If Bounding box is not given, get a tight bounding box from mask image
        bbox = (
            np.array(img_dict["bbox"])
            if img_dict["bbox"]
            else get_tight_bbox(mask_image)
        )

        img_shape = mask_image.shape
        # bbox[0] *= img_shape[1]
        # bbox[2] *= img_shape[1]
        # bbox[1] *= img_shape[0]
        # bbox[3] *= img_shape[0]
        bbox = bbox.astype(int)

        if len(mask_image.shape) == 3:
            mask_image = mask_image[..., -1]

        label_img = mask_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]

        rgb_img = (
            raw_rgb_img[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
            * label_img.unsqueeze(-1).numpy()
        )

        rgb_img_proc = self.inp_transforms(
            PIL.Image.fromarray(rgb_img.astype(np.uint8))
        )

        label_rgb_img_proc = self.label_transforms(
            PIL.Image.fromarray(rgb_img.astype(np.uint8))
        )
        label_rgb_img_proc = label_rgb_img_proc.permute(1, 2, 0)
        label_rgb_img_proc = label_rgb_img_proc.reshape(-1, 3)

        label_img_proc = self.label_transforms(
            PIL.Image.fromarray((255.0 * label_img).numpy().astype(np.uint8))
        )
        label_img_proc = label_img_proc.view(-1).float()

        # Simplify mask path to use it as key for hash and storing camera weights.
        mask_path = mask_path.replace("/", "")
        mask_path = mask_path.replace(".", "")

        return {
            "rgb_img": rgb_img_proc.float(),
            "label_img_path": mask_path,
            "label_rgb_img": label_rgb_img_proc.float(),
            "label_mask_img": label_img_proc,
            "orig_img_path": img_path,
            "mesh_path": "some_placeholder_mesh.off",
        }
        return ret_dict


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


class GenericImgMaskModule(LightningDataModule):
    def __init__(self, data_cfg, render_cfg, num_workers=0, debug_mode=False, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.render_cfg = render_cfg
        self.num_workers = num_workers

        self.train_split = read_paths_and_boxes(
            self.data_cfg.train_dataset_file, self.data_cfg
        )
        self.val_split = read_paths_and_boxes(
            self.data_cfg.val_dataset_file, self.data_cfg
        )

    def train_dataloader(self):

        assert self.render_cfg.cam_num > 0, "camera number cannot be 0"

        train_ds = GenericImgMaskDataset(
            self.train_split,
            self.data_cfg,
            self.render_cfg,
        )

        sampler = None

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(train_ds)

        return torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.data_cfg.bs_train,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=sampler is None,
        )

    def val_dataloader(self):

        assert self.render_cfg.cam_num > 0, "camera number cannot be 0"

        val_ds = DatasetPermutationWrapper(
            GenericImgMaskDataset(
                self.val_split,
                self.data_cfg,
                self.render_cfg,
            )
        )
        return torch.utils.data.DataLoader(
            val_ds, batch_size=self.data_cfg.bs_val, num_workers=self.num_workers
        )

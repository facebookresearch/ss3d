from __future__ import absolute_import, division, print_function

import os
import os.path as osp
import pdb
import time

import hydra
import numpy as np
import submitit
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as T_F
from data.distillation import DistillationDataModule, get_paths
from hydra_config import config
from model import get_model
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler
from torchvision.transforms import ToTensor
from volumetric_render import (
    network_query_fn_validation,
    network_query_fn_train,
    render_img,
    render_rays,
    get_rays_from_angles,
)


os.environ["MKL_THREADING_LAYER"] = "GNU"

_curr_path = osp.dirname(osp.abspath(__file__))
_base_path = _curr_path  # osp.join(_curr_path, "..")


class VolumetricNetworkCkpt(LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg

        # make the e2e (encoder + decoder) model.
        self.model = get_model(cfg.model)

        # Save hyperparameters
        self.save_hyperparameters(cfg)

        cam_num = self.cfg.render.cam_num if self.cfg.render.cam_num > 0 else 1
        self.ray_num = cam_num * self.cfg.render.ray_num_per_cam


def create_encoder_transforms():

    transforms = T.Compose(
        [
            # T.Pad(padding=),
            T.RandomApply(
                torch.nn.ModuleList(
                    [T.ColorJitter(brightness=0.15, hue=0.15, saturation=0.15)]
                ),
                p=0.5,
            ),
            T.RandomApply(
                torch.nn.ModuleList([T.GaussianBlur(kernel_size=(3, 3))]),
                p=0.5,
            ),
            T.RandomRotation(degrees=(-45, 45)),
            # T.RandomAdjustSharpness(sharpness_factor=2),
            T.RandomHorizontalFlip(p=0.5),
            T.Resize((224, 224)),
            # T.RandomVerticalFlip(p=0.5),
        ]
    )
    return transforms


class VolumetricNetwork(LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg

        # make the e2e (encoder + decoder) model.
        self.model = get_model(cfg.model)

        # Save hyperparameters
        self.save_hyperparameters(cfg)

        cam_num = self.cfg.render.cam_num if self.cfg.render.cam_num > 0 else 1
        self.ray_num = cam_num * self.cfg.render.ray_num_per_cam

        self.load_ckpts_on_cpu()

        self.encoder_transforms = create_encoder_transforms()
        self.use_encoder_transforms = self.cfg.distillation.use_encoder_transforms

    def _process_encoder_images(self, inp_images):
        if self.use_encoder_transforms:
            out_enc_imgs = []
            for img in inp_images:
                temp_img = T_F.pad(img, torch.randint(0, 15, (1,)).item())
                temp_img = self.encoder_transforms(temp_img)
                out_enc_imgs.append(temp_img)
            inp_images = torch.stack(out_enc_imgs, dim=0)
        return inp_images

    def load_ckpts_on_cpu(self):

        self.distillation_ckpts = []

        assert self.cfg.distillation.ckpts_root_dir is not None

        checkpoint_paths = get_paths(
            self.cfg.distillation.ckpts_root_dir,
            self.cfg.distillation.regex_match,
            self.cfg.distillation.regex_exclude,
        )
        checkpoint_paths = [
            self.cfg.distillation.warehouse3d_ckpt_path
        ] + checkpoint_paths
        print("!!!!!!!!!!!!!Loading checkpoints from root dir!!!!!!!!")
        print(checkpoint_paths, sep="\n")
        ckpt_idx = 0
        for checkpoint_path in checkpoint_paths:
            temp_model = VolumetricNetworkCkpt.load_from_checkpoint(checkpoint_path)
            temp_model = temp_model.to("cpu")
            self.distillation_ckpts.append(temp_model)
            out_str = f"loading checkpoint {ckpt_idx} onto cpu for distillation"
            tot_m, used_m, free_m = map(
                int, os.popen("free -t -m").readlines()[-1].split()[1:]
            )
            out_str += f" Memory Stats: Total {tot_m}, Used {used_m}, free {free_m}"
            ckpt_idx += 1
            print(out_str)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optim.lr)
        if self.cfg.optim.use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=0.5,
                patience=75,
                threshold_mode="abs",
                threshold=0.005,
            )
            scheduler = {"scheduler": scheduler, "monitor": "train_loss"}
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer

    def validation_step(self, batch, batch_idx):
        """
        This is the method that gets distributed
        """
        with torch.no_grad():

            if self.use_encoder_transforms:
                enc_inp_images = self._process_encoder_images(batch["rgb_img"])
            else:
                enc_inp_images = batch["rgb_img"]

            if self.cfg.distillation.mode == "point":
                loss = self.compute_point_loss(batch)
            else:
                loss = self.compute_ray_loss(batch)

            # Vol Render output image for logging
            def render_ref_pose_images(model, inp_img):
                c_latent = model.encoder(inp_img.unsqueeze(0))
                render_kwargs = {
                    "network_query_fn": network_query_fn_validation,
                    "N_samples": 100,
                    "decoder": model.decoder,
                    "c_latent": c_latent,
                    "chunk": 1000,
                    "device": self.device,
                    "has_rgb": self.cfg.render.rgb,
                    "has_normal": False,
                }

                poses = [
                    (0, 0, 1.5),
                    (90, 90, 1.5),
                    (0, 90, 1.5),
                    (90, 0, 1.5),
                ]
                occ_imgs = []
                rgb_imgs = []
                for p in poses:
                    elev_angle, azim_angle, dist = p
                    _, occ_img, rgb_img, _ = render_img(
                        dist=torch.tensor([dist]).type_as(batch["dist"][-1]),
                        elev_angle=torch.tensor([elev_angle]).type_as(
                            batch["dist"][-1]
                        ),
                        azim_angle=torch.tensor([azim_angle]).type_as(
                            batch["dist"][-1]
                        ),
                        img_size=self.cfg.render.img_size,
                        focal=self.cfg.render.focal_length,
                        render_kwargs=render_kwargs,
                    )
                    occ_imgs.append(occ_img)
                    rgb_imgs.append(rgb_img)

                occ_img = torch.cat(occ_imgs, dim=0)
                if self.cfg.render.rgb:
                    rgb_img = (torch.cat(rgb_imgs, dim=0)).permute(2, 0, 1)
                else:
                    rgb_img = None

                return rgb_img, occ_img.unsqueeze(0)

            rgb_img, occ_img = render_ref_pose_images(self.model, batch["rgb_img"][-1])

            model_id = batch["dataset_id"][-1]
            temp_model = self.distillation_ckpts[model_id.item()]

            temp_model = temp_model.to(self.device)

            rgb_img_label, occ_img_label = render_ref_pose_images(
                temp_model.model, batch["rgb_img"][-1]
            )
            self.distillation_ckpts[model_id.item()] = temp_model.to("cpu")

        return {
            "loss": loss,
            "inp_img": batch["rgb_img"][-1],
            "mask_teacher": occ_img_label,
            "rgb_teacher": rgb_img_label,
            "vol_render": occ_img,
            "vol_render_rgb": rgb_img,
            "transformed_encoder_inp_img": enc_inp_images[-1],
        }

    def validation_epoch_end(self, validation_epoch_outputs):

        avg_loss = torch.cat(
            [l["loss"].unsqueeze(0) for l in validation_epoch_outputs]
        ).mean()

        inp_img = torch.cat([l["inp_img"] for l in validation_epoch_outputs], -1)
        self.logger.experiment.add_image("val_inp_rgb", inp_img, self.global_step)

        inp_img = torch.cat(
            [l["transformed_encoder_inp_img"] for l in validation_epoch_outputs], -1
        )
        self.logger.experiment.add_image(
            "transformed_encoder_inp_img", inp_img, self.global_step
        )

        vol_render = torch.cat([l["vol_render"] for l in validation_epoch_outputs], -1)
        self.logger.experiment.add_image("val_vol_render", vol_render, self.global_step)

        mask_teacher = torch.cat(
            [l["mask_teacher"] for l in validation_epoch_outputs], -1
        )
        self.logger.experiment.add_image("mask_teacher", mask_teacher, self.global_step)

        if self.cfg.render.rgb:
            vol_render_rgb = torch.cat(
                [l["vol_render_rgb"] for l in validation_epoch_outputs], -1
            )
            self.logger.experiment.add_image(
                "vol_render_rgb", vol_render_rgb, self.global_step
            )

            rgb_teacher = torch.cat(
                [l["rgb_teacher"] for l in validation_epoch_outputs], -1
            )
            self.logger.experiment.add_image(
                "rgb_teacher", rgb_teacher, self.global_step
            )

        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": avg_loss, "progress_bar": {"global_step": self.global_step}}

    def _extract_point_batch(self, batch):

        inp_imgs = batch["rgb_img"]
        model_ids = batch["dataset_id"]

        batch_size = len(model_ids)
        num_points = self.cfg.distillation.num_points  # TODO: Tune This!
        sample_bounds_min = self.cfg.distillation.sample_bounds[0]
        sample_bounds_max = self.cfg.distillation.sample_bounds[1]

        with torch.no_grad():
            query_points = torch.rand(batch_size, num_points, 3)
            query_points = (
                sample_bounds_min
                + (sample_bounds_max - sample_bounds_min) * query_points
            )
            query_points = query_points.to(self.device)
            # outputs are r,g,b,density
            # TODO: Still not clear how to handle density!!
            outputs = []

            for i, model_id in enumerate(model_ids):

                temp_model = self.distillation_ckpts[model_id.item()]
                temp_model = temp_model.to(self.device)
                temp_c_latent = temp_model.model.encoder(inp_imgs[i].unsqueeze(0))

                outputs.append(
                    network_query_fn_validation(
                        query_points[i].unsqueeze(0),
                        temp_c_latent,
                        temp_model.model.decoder,
                    )
                )
                self.distillation_ckpts[model_id.item()] = temp_model.to("cpu")

        outputs = torch.cat(outputs, dim=0)
        return inp_imgs, query_points, outputs

    def compute_point_loss(self, batch):

        inp_imgs, query_points, labels_raw = self._extract_point_batch(batch)

        if self.use_encoder_transforms:
            inp_imgs = self._process_encoder_images(inp_imgs)

        self.model.to(self.device)
        c_latent = self.model.encoder(inp_imgs)
        preds_raw = network_query_fn_train(query_points, c_latent, self.model.decoder)

        colors = preds_raw[..., :3]
        colors_labels = labels_raw[..., :3]
        loss_rgb = torch.nn.functional.mse_loss(colors, colors_labels)

        density = torch.nn.functional.relu(preds_raw[..., 3])
        density_labels = torch.nn.functional.relu(labels_raw[..., 3])
        loss_density = torch.nn.functional.mse_loss(density, density_labels)

        loss = (loss_density + loss_rgb) / 2.0

        return loss

    def _extract_ray_batch(self, batch):

        inp_imgs = batch["rgb_img"]
        model_ids = batch["dataset_id"]
        batch_size = len(model_ids)

        with torch.no_grad():
            rays = get_rays_from_angles(
                H=self.cfg.render.img_size,
                W=self.cfg.render.img_size,
                focal=float(self.cfg.render.focal_length),
                near_plane=self.cfg.render.near_plane,
                far_plane=self.cfg.render.far_plane,
                elev_angles=batch["elev_angle"],
                azim_angles=batch["azim_angle"],
                dists=batch["dist"],
                device=self.device,
                indices=batch["flat_indices"],
            )  # [(N*Num_rays), 8] #2d

            mask_ray_labels = []
            rgb_ray_labels = []

            rays = rays.reshape(batch_size, self.ray_num, rays.shape[-1])

            for i, model_id in enumerate(model_ids):

                temp_model = self.distillation_ckpts[model_id.item()]
                temp_model = temp_model.to(self.device)
                temp_c_latent = temp_model.model.encoder(inp_imgs[i].unsqueeze(0))
                temp_c_latent = temp_c_latent.repeat(self.ray_num, 1)

                ray_outs = render_rays(
                    ray_batch=rays[i],
                    c_latent=temp_c_latent,
                    decoder=temp_model.model.decoder,
                    N_samples=self.cfg.render.on_ray_num_samples,
                    has_rgb=self.cfg.render.rgb,
                    has_normal=self.cfg.render.normals,
                )
                mask_ray_labels.append(ray_outs["acc_map"])
                rgb_ray_labels.append(ray_outs["rgb_map"])

                self.distillation_ckpts[model_id.item()] = temp_model.to("cpu")

        rays = rays.reshape(-1, rays.shape[-1])
        mask_ray_labels = torch.cat(mask_ray_labels, dim=0)
        rgb_ray_labels = torch.cat(rgb_ray_labels, dim=0)
        return inp_imgs, mask_ray_labels, rgb_ray_labels, rays

    def compute_ray_loss(self, batch):

        # [N,3, img_size, img_size], 1-d [N*num_rays], [N*num_rays, 8]
        (
            inp_imgs,
            mask_ray_labels,
            rgb_ray_labels,
            rays,
        ) = self._extract_ray_batch(batch)

        if self.use_encoder_transforms:
            inp_imgs = self._process_encoder_images(inp_imgs)

        self.model.to(self.device)
        c_latent = self.model.encoder(inp_imgs)  # [N, c_dim]
        # For instance, C = [[1,2],[3,4]] and num_rays = 2
        # below lines return C = [[1,2],[1,2],[3,4],[3,4]]
        c_latent = (
            c_latent.unsqueeze(1).repeat(1, self.ray_num, 1).view(-1, c_latent.shape[1])
        )  # [N*N_num_rays, c_dim]

        ray_outs = render_rays(
            ray_batch=rays,  # [N*num_rays, 8]
            c_latent=c_latent,  # [N*N_num_rays, c_dim]
            decoder=self.model.decoder,  # nn.Module
            N_samples=self.cfg.render.on_ray_num_samples,  # int
            has_rgb=self.cfg.render.rgb,
            has_normal=self.cfg.render.normals,
        )
        mask_ray_outs = ray_outs["acc_map"].to(self.device)  # [N*num_rays] 1-d

        loss = []
        loss_mask = torch.nn.functional.mse_loss(
            mask_ray_labels,
            mask_ray_outs,  # reduction="none"
        )  # [N*num_rays] 1-d
        loss.append(loss_mask)

        if self.cfg.render.rgb:
            rgb_ray_outs = ray_outs["rgb_map"].to(self.device)
            loss_rgb = torch.nn.functional.mse_loss(
                rgb_ray_labels,
                rgb_ray_outs,
            )
            loss.append(loss_rgb)

        # loss = torch.tensor(loss, requires_grad=True)
        if self.cfg.render.rgb:
            loss = (loss_mask + loss_rgb) / 2.0
        else:
            loss = loss_mask

        return loss

    def training_step(self, batch, batch_idx):
        """
        This is the method that gets distributed
        """
        if self.cfg.distillation.mode == "point":
            loss = self.compute_point_loss(batch)
        else:
            loss = self.compute_ray_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss, "progress_bar": {"global_step": self.global_step}}

    def forward(self, mode, inputs):
        pass


def train_model(cfg):
    print(OmegaConf.to_yaml(cfg))  # TODO: Add this to tensorboard logging

    # configure data loader
    distillation_data_loader = DistillationDataModule(
        cfg=cfg,
        base_path=_base_path,
    )

    model = VolumetricNetwork(cfg=cfg)
    log_dir = osp.join(_base_path, cfg.logging.log_dir, cfg.logging.name)
    os.makedirs(log_dir, exist_ok=True)
    OmegaConf.save(cfg, osp.join(log_dir, "config.txt"))

    logger = TensorBoardLogger(
        osp.join(_base_path, cfg.logging.log_dir), name=cfg.logging.name
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_val_epochs=cfg.optim.save_freq,
        filename="checkpoint_{epoch}",
    )

    if cfg.optim.use_pretrain:
        temp_model = VolumetricNetworkCkpt.load_from_checkpoint(
            cfg.optim.checkpoint_path
        )
        model.model.load_state_dict(temp_model.model.state_dict())
    else:
        checkpoint = None

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # https://pytorch-lightning.readthedocs.io/en/latest/trainer.html
    trainer = Trainer(
        logger=logger,
        gpus=cfg.resources.gpus,
        num_nodes=cfg.resources.num_nodes,
        val_check_interval=cfg.optim.val_check_interval,
        limit_val_batches=cfg.optim.num_val_iter,
        callbacks=[checkpoint_callback, lr_monitor],
        # resume_from_checkpoint=checkpoint,
        resume_from_checkpoint=None,  # Only loading weights
        max_epochs=cfg.optim.max_epochs,
        accelerator=cfg.resources.accelerator
        if cfg.resources.accelerator != "none"
        else None,
        deterministic=False,
        # profiler="simple",
    )
    trainer.fit(model, distillation_data_loader)


@hydra.main(config_name="config")
def main(cfg: config.Config) -> None:
    # Set the everythin - randon, numpy, torch, torch manula, cuda!!
    seed_everything(12)

    # If not cluster launch job locally
    if not cfg.resources.use_cluster:
        train_model(cfg)
    else:
        print(OmegaConf.to_yaml(cfg))  # TODO: Add this to tensorboard logging

        # dummy to get veriosn only
        dummy_logger = TensorBoardLogger(
            osp.join(_base_path, cfg.logging.log_dir), name=cfg.logging.name
        )

        submitit_dir = osp.join(
            _base_path,
            cfg.logging.log_dir,
            cfg.logging.name,
            "submitit_" + str(dummy_logger.version),
        )
        executor = submitit.AutoExecutor(folder=submitit_dir)

        job_kwargs = {
            "mem_gb": 700,
            "timeout_min": cfg.resources.time,
            "name": cfg.logging.name,
            "slurm_partition": cfg.resources.partition,
            "gpus_per_node": cfg.resources.gpus,
            "tasks_per_node": cfg.resources.gpus,  # one task per GPU
            "cpus_per_task": 5,
            "nodes": cfg.resources.num_nodes,
        }
        if cfg.resources.max_mem:
            job_kwargs["slurm_constraint"] = "volta32gb"
        if cfg.resources.partition == "priority":
            job_kwargs["slurm_comment"] = cfg.resources.comment

        executor.update_parameters(**job_kwargs)
        job = executor.submit(train_model, cfg)
        print("Submitit Job ID:", job.job_id)  # ID of your job


if __name__ == "__main__":
    main()

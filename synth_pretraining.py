from __future__ import absolute_import, division, print_function

import os
import os.path as osp
import pdb

import hydra
import matplotlib.pyplot as plt
import numpy as np
import submitit
import torch
from data.warehouse3d import (
    WareHouse3DModule,
    extract_data_train,
)
from hydra_config import config
from model import get_model
from mpl_toolkits.mplot3d import axes3d
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler
from torchvision.transforms import ToTensor
from volumetric_render import (
    network_query_fn_validation,
    render_img,
    render_rays,
)


os.environ["MKL_THREADING_LAYER"] = "GNU"

_curr_path = osp.dirname(osp.abspath(__file__))
_base_path = _curr_path  # osp.join(_curr_path, "..")


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

        # Matplotlib figures
        self.iou_fig, self.iou_ax = plt.subplots()
        self.IOU_THRESHOLDS = np.linspace(0, 100, 200).tolist()

        self.voxels_fig = plt.figure()
        self.voxels_ax = self.voxels_fig.add_subplot(111, projection="3d")

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

    def extract_batch_input(self, batch):
        return extract_data_train(batch, self.cfg.render, self.device)

    def validation_step(self, batch, batch_idx):
        """
        This is the method that gets distributed
        """
        with torch.no_grad():

            (
                inp_imgs,
                mask_ray_labels,
                rgb_label_rays,
                normal_ray_labels,
                depth_ray_labels,
                rays,
            ) = self.extract_batch_input(batch)

            c_latent = self.model.encoder(inp_imgs)
            c_latent = (
                c_latent.unsqueeze(1)
                .repeat(1, self.ray_num, 1)
                .view(-1, c_latent.shape[1])
            )

            ray_outs = render_rays(
                ray_batch=rays,
                c_latent=c_latent,
                decoder=self.model.decoder,
                N_samples=self.cfg.render.on_ray_num_samples,
                has_rgb=self.cfg.render.rgb,
            )
            mask_ray_outs = ray_outs["acc_map"].to(self.device)

            loss = torch.nn.functional.mse_loss(
                mask_ray_labels,
                mask_ray_outs,  # reduction="none"
            )

            # Vol Render output image for logging
            render_kwargs = {
                "network_query_fn": network_query_fn_validation,
                "N_samples": 100,
                "decoder": self.model.decoder,
                "c_latent": c_latent[-1].reshape(1, -1),
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
                    elev_angle=torch.tensor([elev_angle]).type_as(batch["dist"][-1]),
                    azim_angle=torch.tensor([azim_angle]).type_as(batch["dist"][-1]),
                    img_size=self.cfg.render.img_size,
                    focal=self.cfg.render.focal_length,
                    render_kwargs=render_kwargs,
                )
                occ_imgs.append(occ_img)
                rgb_imgs.append(rgb_img)

            occ_img = torch.cat(occ_imgs, dim=0)

            rgb_img = (torch.cat(rgb_imgs, dim=0)).permute(2, 0, 1)

        return {
            "loss": loss,
            "inp_img": inp_imgs[-1],
            "rgb_gt": None,
            "normal_gt": None,
            "vol_render": occ_img.unsqueeze(0),
            "vol_render_rgb": rgb_img,
            "vol_render_normal": None,
        }

    def validation_epoch_end(self, validation_epoch_outputs):

        avg_loss = torch.cat(
            [l["loss"].unsqueeze(0) for l in validation_epoch_outputs]
        ).mean()

        # Input Image
        inp_img = torch.cat([l["inp_img"] for l in validation_epoch_outputs], -1)
        self.logger.experiment.add_image("val_inp_rgb", inp_img, self.global_step)

        # Mask Rendering
        vol_render = torch.cat([l["vol_render"] for l in validation_epoch_outputs], -1)
        self.logger.experiment.add_image("val_vol_render", vol_render, self.global_step)

        # RGB Rendering
        vol_render_rgb = torch.cat(
            [l["vol_render_rgb"] for l in validation_epoch_outputs], -1
        )
        self.logger.experiment.add_image(
            "vol_render_rgb", vol_render_rgb, self.global_step
        )

        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": avg_loss, "progress_bar": {"global_step": self.global_step}}

    def training_step(self, batch, batch_idx):
        """
        This is the method that gets distributed
        """
        # [N,3, img_size, img_size], 1-d [N*num_rays], [N*num_rays, 8]
        (
            inp_imgs,
            mask_ray_labels,
            rgb_ray_labels,
            normal_ray_labels,
            depth_ray_labels,
            rays,
        ) = self.extract_batch_input(batch)

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

        if self.cfg.render.normals:
            normal_ray_outs = ray_outs["normal_map"].to(self.device)
            normal_ray_outs = normal_ray_outs.reshape(-1, 3)
            normal_ray_labels = normal_ray_labels.reshape(-1, 3)
            loss_normal = normal_ray_outs * normal_ray_labels
            loss_normal = torch.abs(loss_normal.sum(-1))

            loss_normal = 1.0 - loss_normal
            loss_normal = loss_normal.mean()
            loss.append(loss_normal)

        # loss = torch.tensor(loss, requires_grad=True)
        if self.cfg.render.rgb:
            loss = (loss_mask + loss_rgb) / 2.0
        else:
            loss = loss_mask
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss, "progress_bar": {"global_step": self.global_step}}

    def forward(self, mode, inputs):
        pass


def train_model(cfg):
    print(OmegaConf.to_yaml(cfg))

    # configure data loader
    data_module = WareHouse3DModule(
        data_cfg=cfg.data,
        render_cfg=cfg.render,
        num_workers=cfg.resources.num_workers,
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
        temp_model = VolumetricNetwork.load_from_checkpoint(cfg.optim.checkpoint_path)
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
        resume_from_checkpoint=None,  # Only loading weights
        max_epochs=cfg.optim.max_epochs,
        accelerator=cfg.resources.accelerator
        if cfg.resources.accelerator != "none"
        else None,
        deterministic=False,
    )
    trainer.fit(model, data_module)


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

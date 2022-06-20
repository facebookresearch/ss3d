from __future__ import absolute_import, division, print_function

import copy
import os
import os.path as osp
import pdb

import hydra
import numpy as np
import submitit
import torch
from data.generic_img_mask_loader import GenericImgMaskModule
from data.warehouse3d import (
    get_rays_multiplex,
    extract_data_train,
)
from hydra_config import config
from model import get_model
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler
from torchvision.transforms import ToTensor
from tqdm import tqdm
from volumetric_render import (
    network_query_fn_validation,
    render_img,
    render_rays,
)

os.environ["MKL_THREADING_LAYER"] = "GNU"

_curr_path = osp.dirname(osp.abspath(__file__))
_base_path = _curr_path  # osp.join(_curr_path, "..")


class CameraHandler:
    def __init__(
        self,
        # TODO : learnable principle and theta and wider distance range,
        cfg,
        elev_range=[-90, 90],
        azim_range=[-180, 180],
        dist_range=[1.0, 2.0],
        # optimization related
        num_iters=20,
        optim_lr=0.1,
        # TODO: add more optimization things like camera params etc.
    ):
        self.cfg = cfg

        self.num_cams = cfg.render.cam_num
        self.azim_range = azim_range
        self.elev_range = elev_range
        self.dist_range = dist_range
        self.optim_lr = optim_lr

        # parmeters are stored in this in dist[frame_id or hast] = matrix
        self.weights_path = self._get_weights_path()
        self.cameras = {}
        self.linear_weights = {}

        self.num_iters = num_iters
        assert self.cfg.render.cam_num > 0, "Cam number should be > 0 for no camera"
        self.ray_num = self.cfg.render.cam_num * self.cfg.render.ray_num_per_cam
        self.shape_reg = None

    def _get_weights_path(self):
        # dummy to get veriosn only
        dummy_logger = TensorBoardLogger(
            osp.join(_base_path, self.cfg.logging.log_dir), name=self.cfg.logging.name
        )

        return osp.join(
            _base_path,
            self.cfg.logging.log_dir,
            self.cfg.logging.name,
            "cameras_" + str(dummy_logger.version),
        )

    def get_cameras(self, frame_list, device):
        # fame_list: list of stirngs  or list of hashes
        cameras = []
        for frame in frame_list:
            temp_path = os.path.splitext(frame[1:])[0]
            temp_path = os.path.join(self.weights_path, temp_path)
            temp_file = os.path.join(temp_path, "cameras.npy")

            if os.path.exists(temp_file):
                try:
                    cam = np.load(temp_file, allow_pickle=True)
                    cam = torch.from_numpy(cam)
                    load_from_memory_flag = True
                except:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("Error loading cameras for frame {}".format(frame))
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    load_from_memory_flag = False
            else:
                load_from_memory_flag = False

            if not load_from_memory_flag:
                cam = []
                azim_init_angles = 10.0 * np.arctanh(
                    np.linspace(-0.1, 0.1, num=self.cfg.render.cam_num, endpoint=False)
                )
                azim_init_angles = azim_init_angles.tolist()
                for i in range(self.cfg.render.cam_num):
                    elev_param = 2 * torch.rand(1) - 1.0  # [-90,90]
                    # azim_param = 2 * (2 * torch.rand(1) - 1.0)  # [-180,180]
                    azim_param = torch.tensor([azim_init_angles[i]])
                    dist = self.dist_range[0] + (
                        self.dist_range[1] - self.dist_range[0]
                    ) * torch.rand(1)
                    cam.append(torch.cat([elev_param, azim_param, dist], dim=0))
                cam = torch.stack(cam, dim=0)
            cameras.append(cam)
        return torch.stack(cameras).to(device)

    def _get_cam_weights(self, frame_list, device):
        camera_weights = []
        for frame in frame_list:
            temp_path = os.path.splitext(frame[1:])[0]
            temp_path = os.path.join(self.weights_path, temp_path)
            temp_file = os.path.join(temp_path, "cam_weights.npy")

            if os.path.exists(temp_file):
                try:
                    cam_weight = np.load(temp_file, allow_pickle=True)
                    cam_weight = torch.from_numpy(cam_weight)
                    load_from_memory_flag = True
                except:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(
                        "Error loading camer linear weights for frame {}".format(frame)
                    )
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    load_from_memory_flag = False
            else:
                load_from_memory_flag = False

            if not load_from_memory_flag:
                cam_weight = torch.rand(self.cfg.render.cam_num)
            camera_weights.append(cam_weight)

        return torch.stack(camera_weights).to(device)

    def _update_cam_weights(self, frame_list, camera_weights):
        for i, frame in enumerate(frame_list):
            temp_path = os.path.splitext(frame[1:])[0]
            temp_path = os.path.join(self.weights_path, temp_path)
            if not os.path.isdir(temp_path):
                os.makedirs(temp_path, exist_ok=True)
            temp_file = os.path.join(temp_path, "cam_weights.npy")
            np.save(temp_file, camera_weights[i].detach().cpu().numpy())

    def _update_cameras(self, frame_list, camera_params):
        for i, frame in enumerate(frame_list):
            temp_path = os.path.splitext(frame[1:])[0]
            temp_path = os.path.join(self.weights_path, temp_path)
            if not os.path.isdir(temp_path):
                os.makedirs(temp_path, exist_ok=True)
            temp_file = os.path.join(temp_path, "cameras.npy")
            np.save(temp_file, camera_params[i].detach().cpu().numpy())

    def _compute_camera_loss(
        self,
        frame_list,
        rgb_imgs,
        mask_imgs,
        cameras,
        decoder,
        c_latent,
        device,
        shape_regularizer=None,
    ):

        mask_ray_labels, rgb_ray_labels, rays = get_rays_multiplex(
            cameras, rgb_imgs, mask_imgs, self.cfg.render, device
        )
        # print("\n")
        # print(mask_ray_labels.shape, rgb_ray_labels.shape,rays.shape)
        # print(rgb_ray_labels.max(), rgb_ray_labels.min(), mask_ray_labels.max(), mask_ray_labels.min())
        # print(rgb_imgs.shape, mask_imgs.shape)
        # print(rgb_imgs.max(), rgb_imgs.min(), mask_imgs.max(), mask_imgs.min())
        # print("\n")

        ray_outs = render_rays(
            ray_batch=rays,  # [N*num_rays, 8]
            c_latent=c_latent,  # [N*N_num_rays, c_dim]
            decoder=decoder,  # nn.Module
            N_samples=self.cfg.render.on_ray_num_samples,  # int
            has_rgb=self.cfg.render.rgb,
            has_normal=self.cfg.render.normals,
            retraw=shape_regularizer is not None,
        )
        mask_ray_outs = ray_outs["acc_map"].to(device)  # [N*num_rays] 1-d

        loss_mask = torch.nn.functional.mse_loss(
            mask_ray_labels, mask_ray_outs, reduction="none"
        )  # [N*num_rays] 1-d
        loss_mask = loss_mask.reshape(
            len(frame_list), self.cfg.render.cam_num, self.cfg.render.ray_num_per_cam
        )
        loss_mask = loss_mask.mean(-1)

        if self.cfg.render.rgb:
            rgb_ray_outs = ray_outs["rgb_map"].to(device)
            loss_rgb = torch.nn.functional.mse_loss(
                rgb_ray_labels, rgb_ray_outs, reduction="none"
            )
            loss_rgb = loss_rgb.reshape(
                len(frame_list),
                self.cfg.render.cam_num,
                self.cfg.render.ray_num_per_cam,
                3,
            )
            loss_rgb = loss_rgb.mean((-1, -2))

        if self.cfg.render.rgb:
            # loss = (loss_mask + loss_rgb) / 2.0
            loss = 0.2 * loss_mask + 0.8 * loss_rgb
        else:
            loss = loss_mask

        if shape_regularizer is not None:
            reg_outs = torch.sigmoid(shape_regularizer(ray_outs["points"]))
            reg_outs = reg_outs.squeeze(-1)
            instance_outs = torch.sigmoid(ray_outs["raw"][..., 3])

            reg_loss = torch.nn.functional.mse_loss(reg_outs, instance_outs)
            return loss, reg_loss
        else:
            return loss

    def _softmin_loss(
        self,
        frame_list,
        rgb_imgs,
        mask_imgs,
        cameras,
        decoder,
        c_latent_from_encoder,
        device,
        shape_reg=None,
    ):
        # compute decoder loss
        cameras = cameras.detach()
        decoder.zero_grad()
        c_latent = (
            c_latent_from_encoder.unsqueeze(1)
            .repeat(1, self.ray_num, 1)
            .view(-1, c_latent_from_encoder.shape[1])
        )  # [N*N_num_rays, c_dim]

        loss = self._compute_camera_loss(
            frame_list, rgb_imgs, mask_imgs, cameras, decoder, c_latent, device
        )

        loss_softmin = torch.nn.functional.softmin(
            loss * self.cfg.render.softmin_temp
        ).detach()  # detatch?? # this will moving average!!!
        loss = torch.mul(loss_softmin, loss)
        return loss.mean()

    def _softmax_loss(
        self,
        frame_list,
        rgb_imgs,
        mask_imgs,
        cameras,
        decoder,
        c_latent_from_encoder,
        device,
        n_iter=10,  # TODO: tune this!
        shape_reg=None,
    ):
        # get weights
        camera_weights = self._get_cam_weights(frame_list, device)
        camera_weights = torch.nn.Parameter(camera_weights)

        # create optimizer
        cam_weight_optimizer = torch.optim.Adam(
            [camera_weights], lr=self.optim_lr  # TODO: change this!
        )

        cameras = cameras.detach()
        c_latent = c_latent_from_encoder.detach()
        c_latent = (
            c_latent.unsqueeze(1).repeat(1, self.ray_num, 1).view(-1, c_latent.shape[1])
        )  # [N*N_num_rays, c_dim]

        for i in range(n_iter):
            cam_weight_optimizer.zero_grad()
            decoder.zero_grad()

            loss = self._compute_camera_loss(
                frame_list, rgb_imgs, mask_imgs, cameras, decoder, c_latent, device
            )

            weights_softmax = torch.nn.functional.softmax(
                camera_weights
            )  # TODO: temperature?
            loss = torch.mul(weights_softmax, loss)
            loss = loss.mean()

            loss.backward()
            cam_weight_optimizer.step()
            cam_weight_optimizer.zero_grad()
            decoder.zero_grad()

        # update weights
        self._update_cam_weights(frame_list, camera_weights)

        # compute model loss!
        camera_weights = camera_weights.detach()
        c_latent = (
            c_latent_from_encoder.unsqueeze(1)
            .repeat(1, self.ray_num, 1)
            .view(-1, c_latent_from_encoder.shape[1])
        )  # [N*N_num_rays, c_dim]

        if shape_reg is not None:
            loss, reg_loss = self._compute_camera_loss(
                frame_list,
                rgb_imgs,
                mask_imgs,
                cameras,
                decoder,
                c_latent,
                device,
                shape_regularizer=shape_reg,
            )
        else:
            loss = self._compute_camera_loss(
                frame_list,
                rgb_imgs,
                mask_imgs,
                cameras,
                decoder,
                c_latent,
                device,
            )
        weights_softmax = torch.nn.functional.softmax(camera_weights)
        loss = torch.mul(weights_softmax, loss)
        loss = loss.sum()

        if shape_reg is not None:
            return loss, reg_loss
        else:
            return loss

    def val_get_bet_camera(self, batch_dict, device):
        frame_list = batch_dict["label_img_path"]  # .detach().cpu()
        camera_params = self.get_cameras(frame_list, device)
        elev_angles = torch.nn.Parameter(camera_params[..., 0])
        azim_angles = torch.nn.Parameter(camera_params[..., 1])
        dists = torch.nn.Parameter(camera_params[..., 2])
        cameras = torch.empty(len(frame_list), self.cfg.render.cam_num, 3).to(device)
        cameras[..., 0] = torch.tanh(0.1 * elev_angles) * 10 * self.elev_range[1]
        cameras[..., 1] = torch.tanh(0.1 * azim_angles) * 10 * self.azim_range[1]
        cameras[..., 2] = dists
        camera_weights = self._get_cam_weights(frame_list, device)
        weights_softmax = torch.nn.functional.softmax(camera_weights)
        _, inds = torch.max(weights_softmax, dim=1)

        return cameras, weights_softmax, inds

    def optimize_cameras(
        self, batch_dict, c_latent_from_encoder, decoder, device, shape_reg=None
    ):
        # frame_list: (n,)
        # c_latent: (n, c_dim)

        frame_list = batch_dict["label_img_path"]  # .detach().cpu()
        rgb_imgs = batch_dict["label_rgb_img"]
        mask_imgs = batch_dict["label_mask_img"]

        c_latent = c_latent_from_encoder.detach()
        c_latent = (
            c_latent.unsqueeze(1).repeat(1, self.ray_num, 1).view(-1, c_latent.shape[1])
        )  # [N*N_num_rays, c_dim]
        # Define optimizer
        camera_params = self.get_cameras(frame_list, device)
        elev_angles = torch.nn.Parameter(camera_params[..., 0])
        azim_angles = torch.nn.Parameter(camera_params[..., 1])
        dists = torch.nn.Parameter(camera_params[..., 2])

        cam_optimizer = torch.optim.Adam(
            [elev_angles, azim_angles, dists], lr=self.optim_lr
        )

        for i in range(self.num_iters):
            cam_optimizer.zero_grad()
            decoder.zero_grad()

            cameras = torch.empty(len(frame_list), self.cfg.render.cam_num, 3).to(
                device
            )
            cameras[..., 0] = torch.tanh(0.1 * elev_angles) * 10 * self.elev_range[1]
            cameras[..., 1] = torch.tanh(0.1 * azim_angles) * 10 * self.azim_range[1]
            cameras[..., 2] = dists

            loss = self._compute_camera_loss(
                frame_list, rgb_imgs, mask_imgs, cameras, decoder, c_latent, device
            )
            loss_camera = loss.mean()

            loss_camera.backward()
            cam_optimizer.step()
            cam_optimizer.zero_grad()
            decoder.zero_grad()
            if i == 0:
                pre_optim_loss = loss_camera.detach()

        # TODO: Check if camera_params are actually updated!! If not uncomment and fix!
        # camera_params = torch.cat([elev_angles, azim_angles, dists], dim=-1)
        camera_params = camera_params.detach()
        self._update_cameras(frame_list, camera_params)
        cam_optimizer = None

        # compute decoder loss
        cameras = cameras.detach()
        decoder.zero_grad()
        reg_loss = 0.0
        if self.cfg.render.loss_mode == "softmin":
            loss = self._softmin_loss(
                frame_list,
                rgb_imgs,
                mask_imgs,
                cameras,
                decoder,
                c_latent_from_encoder,
                device,
                shape_reg=shape_reg,
            )
        elif self.cfg.render.loss_mode == "softmax":
            if self.shape_reg is not None:
                loss, reg_loss = self._softmax_loss(
                    frame_list,
                    rgb_imgs,
                    mask_imgs,
                    cameras,
                    decoder,
                    c_latent_from_encoder,
                    device,
                    n_iter=1,
                    shape_reg=shape_reg,
                )
            else:
                loss = self._softmax_loss(
                    frame_list,
                    rgb_imgs,
                    mask_imgs,
                    cameras,
                    decoder,
                    c_latent_from_encoder,
                    device,
                    n_iter=1,
                    shape_reg=shape_reg,
                )

        ret_dict = {
            "pre_optim_loss": pre_optim_loss,
            "post_optim_loss": loss_camera,
            "decoder_loss": loss,
            "regularizer_loss": reg_loss,
        }

        return ret_dict


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

        assert self.cfg.render.cam_num > 0, "Cam number should be > 0 for no camera"
        self.ray_num = self.cfg.render.cam_num * self.cfg.render.ray_num_per_cam


class VolumetricNetwork(LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg

        # make the e2e (encoder + decoder) model.
        self.model = get_model(cfg.model)

        self.shape_reg = None

        # Save hyperparameters
        self.save_hyperparameters(cfg)

        assert self.cfg.render.cam_num > 0, "Cam number should be > 0 for no camera"
        self.ray_num = self.cfg.render.cam_num * self.cfg.render.ray_num_per_cam

        # Camera pose handler
        self.camera_handler = CameraHandler(
            self.cfg,
        )
        self.temp = torch.rand(2, 2)

    def configure_optimizers(self):

        fine_tune = self.cfg.model.fine_tune
        if fine_tune == "none":
            params = [self.temp]
        elif fine_tune == "all":
            params = list(self.model.parameters())
        elif fine_tune == "encoder":
            params = list(self.model.encoder.parameters())
        elif fine_tune == "decoder":
            params = list(self.model.decoder.parameters())

        if self.shape_reg is not None:
            params += list(self.shape_reg.parameters())

        return torch.optim.Adam(params, lr=self.cfg.optim.lr)

    def validation_step(self, batch, batch_idx):
        """
        This is the method that gets distributed
        """

        with torch.no_grad():
            label_img = batch["label_rgb_img"][-1]
            label_img = label_img.reshape(128, 128, 3).permute(2, 0, 1)

            label_mask_img = batch["label_mask_img"][-1]
            label_mask_img = label_mask_img.reshape(128, 128)

            inp_imgs = batch["rgb_img"]
            inp_imgs = inp_imgs.to(self.device)

            c_latent = self.model.encoder(inp_imgs)

            loss = torch.tensor(float(0.5)).type_as(c_latent)

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

            cameras, cam_weights, inds = self.camera_handler.val_get_bet_camera(
                batch, self.device
            )

            poses = [
                (45.0, 45.0, cameras[-1, inds[-1], 2]),
                (0.0, 90.0, cameras[-1, inds[-1], 2]),
                (0.0, 0.0, cameras[-1, inds[-1], 2]),
                (90.0, 0.0, cameras[-1, inds[-1], 2]),
            ]
            ref_occ_imgs = []
            ref_rgb_imgs = []
            for p in poses:
                elev_angle, azim_angle, dist = p
                _, occ_img, rgb_img, _ = render_img(
                    dist=torch.tensor([dist]),
                    elev_angle=torch.tensor([elev_angle]),
                    azim_angle=torch.tensor([azim_angle]),
                    img_size=self.cfg.render.img_size,
                    focal=self.cfg.render.focal_length,
                    render_kwargs=render_kwargs,
                )
                ref_occ_imgs.append(occ_img)
                ref_rgb_imgs.append(rgb_img)

            ref_occ_imgs = torch.cat(ref_occ_imgs, dim=0)
            if self.cfg.render.rgb:
                ref_rgb_imgs = (torch.cat(ref_rgb_imgs, dim=0)).permute(2, 0, 1)
            else:
                ref_rgb_imgs = None

            depth_img, occ_img, rgb_img, normal_img = render_img(
                dist=cameras[-1, inds[-1], 2],
                elev_angle=cameras[-1, inds[-1], 0],
                azim_angle=cameras[-1, inds[-1], 1],
                img_size=self.cfg.render.img_size,
                focal=self.cfg.render.focal_length,
                render_kwargs=render_kwargs,
            )

            if rgb_img is not None:
                rgb_img = rgb_img.permute(2, 0, 1)

            if normal_img is not None:
                normal_img = normal_img.permute(2, 0, 1)

        return {
            "loss": loss,
            "inp_img": inp_imgs[-1],
            "mask_gt": label_mask_img.unsqueeze(0),
            "rgb_gt": None,
            "normal_gt": None,
            "vol_render": occ_img.unsqueeze(0),
            "vol_render_rgb": rgb_img,
            "vol_render_normal": normal_img,
            "label_rgb_img": label_img,
            "ref_rgb_imgs": ref_rgb_imgs,
        }

    def validation_epoch_end(self, validation_epoch_outputs):
        avg_loss = torch.cat(
            [l["loss"].unsqueeze(0) for l in validation_epoch_outputs]
        ).mean()

        inp_img = torch.cat([l["inp_img"] for l in validation_epoch_outputs], -1)

        mask_gt = torch.cat([l["mask_gt"] for l in validation_epoch_outputs], -1)
        vol_render = torch.cat([l["vol_render"] for l in validation_epoch_outputs], -1)
        if self.cfg.render.rgb:
            vol_render_rgb = torch.cat(
                [l["vol_render_rgb"] for l in validation_epoch_outputs], -1
            )
            self.logger.experiment.add_image(
                "vol_render_rgb", vol_render_rgb, self.global_step
            )

            ref_rgb_imgs = torch.cat(
                [l["ref_rgb_imgs"] for l in validation_epoch_outputs], -1
            )
            self.logger.experiment.add_image(
                "ref_rgb_imgs", ref_rgb_imgs, self.global_step
            )

            label_img = torch.cat(
                [l["label_rgb_img"] for l in validation_epoch_outputs], -1
            )

            self.logger.experiment.add_image(
                "label rgb img", label_img, self.global_step
            )

        self.logger.experiment.add_image("val_inp_rgb", inp_img, self.global_step)
        self.logger.experiment.add_image("val_vol_render", vol_render, self.global_step)
        self.logger.experiment.add_image(
            "val_mesh_gt_render", mask_gt, self.global_step
        )

        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": avg_loss, "progress_bar": {"global_step": self.global_step}}

    def training_step(self, batch, batch_idx):
        """
        This is the method that gets distributed
        """

        inp_imgs = batch["rgb_img"].to(self.device)
        c_latent = self.model.encoder(inp_imgs)  # [N, c_dim]
        out_dict = self.camera_handler.optimize_cameras(
            batch, c_latent, self.model.decoder, self.device, self.shape_reg
        )

        self.log(
            "train_loss",
            out_dict["decoder_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "camera_post_optim_loss",
            out_dict["post_optim_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "camera_pre_optim_loss",
            out_dict["pre_optim_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if self.shape_reg is not None:
            self.log(
                "train_regularizer_loss",
                out_dict["regularizer_loss"],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            loss_out = (
                0.3 * out_dict["decoder_loss"] + 0.7 * out_dict["regularizer_loss"]
            )
            return loss_out
        else:
            return out_dict["decoder_loss"]

    def forward(self, mode, inputs):
        pass


def train_model(cfg):

    print(OmegaConf.to_yaml(cfg))

    data_module = GenericImgMaskModule(
        data_cfg=cfg.data,
        render_cfg=cfg.render,
        num_workers=cfg.resources.num_workers,
    )

    log_dir = osp.join(_base_path, cfg.logging.log_dir, cfg.logging.name)
    os.makedirs(log_dir, exist_ok=True)
    OmegaConf.save(cfg, osp.join(log_dir, "config.txt"))

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_val_epochs=cfg.optim.save_freq,
        filename="checkpoint_{epoch}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Fine tune only cameras
    logger = TensorBoardLogger(
        osp.join(_base_path, cfg.logging.log_dir), name=cfg.logging.name
    )

    if cfg.optim.stage_one_epochs > 0:
        cfg.model.fine_tune = "none"
        print(cfg)
        model = VolumetricNetwork(cfg=cfg)
        if cfg.optim.use_pretrain:
            temp_model = VolumetricNetwork.load_from_checkpoint(
                cfg.optim.checkpoint_path
            )
            model.model.load_state_dict(temp_model.model.state_dict())
        else:
            checkpoint = None

        trainer = Trainer(
            logger=logger,
            gpus=cfg.resources.gpus,
            num_nodes=cfg.resources.num_nodes,
            val_check_interval=cfg.optim.val_check_interval,
            limit_val_batches=cfg.optim.num_val_iter,
            # checkpoint_callback=checkpoint_callback,
            # resume_from_checkpoint=checkpoint,
            resume_from_checkpoint=None,  # Only loading weights
            max_epochs=cfg.optim.stage_one_epochs,
            accelerator=cfg.resources.accelerator
            if cfg.resources.accelerator != "none"
            else None,
            deterministic=False,
            # profiler="simple",
            callbacks=[lr_monitor],
        )
        trainer.fit(model, data_module)
        cam_weights_path = model.camera_handler.weights_path

    # Fine tune the entire network + cameras
    print("#########################################################################")
    print("###################      Fine Tuning Phase      #########################")
    print("#########################################################################")

    logger = TensorBoardLogger(
        osp.join(_base_path, cfg.logging.log_dir), name=cfg.logging.name
    )

    cfg.model.fine_tune = "all"  # TODO: Make this a config param!

    print(cfg)
    model = VolumetricNetwork(cfg=cfg)
    if cfg.optim.use_pretrain:
        temp_model = VolumetricNetwork.load_from_checkpoint(cfg.optim.checkpoint_path)
        model.model.load_state_dict(temp_model.model.state_dict())
    else:
        checkpoint = None

    # Point the model to use old paths
    if cfg.optim.stage_one_epochs > 0:
        model.camera_handler.weights_path = cam_weights_path

    trainer = Trainer(
        logger=logger,
        gpus=cfg.resources.gpus,
        num_nodes=cfg.resources.num_nodes,
        val_check_interval=cfg.optim.val_check_interval,
        limit_val_batches=cfg.optim.num_val_iter,
        callbacks=[checkpoint_callback, lr_monitor],
        resume_from_checkpoint=None,
        max_epochs=cfg.optim.max_epochs,
        accelerator=cfg.resources.accelerator
        if cfg.resources.accelerator != "none"
        else None,
        deterministic=False,
        # profiler="simple",
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
        print(OmegaConf.to_yaml(cfg))

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

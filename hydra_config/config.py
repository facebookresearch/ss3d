from __future__ import absolute_import, division, print_function

import os.path as osp
import pdb
from dataclasses import dataclass, field
from typing import Any, List

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

cs = ConfigStore.instance()


defaults = [
    "_self_",
    {"logging": "tensorboard"},
    {"model": "vnet"},
    {"optim": "default"},
    {"resources": "default"},
    {"data": "warehouse3d"},
    {"render": "default"},
    {"distillation": "empty"},
    {"distillation/ckpts": []},
]


@dataclass
class DataConfig:
    # name: str = ""
    bs_train: int = 1
    bs_val: int = 1
    num_workers: int = 0


@dataclass
class Warehouse3DData(DataConfig):
    train_split: float = 0.95
    validation_split: float = 0.05
    encoder_dir: str = (
        "/private/home/kalyanv/learning_vision3d/datasets/blender/renders_encoder"
    )
    rgb_dir: str = "/private/home/kalyanv/learning_vision3d/datasets/blender/renders"
    mask_dir: str = "/private/home/kalyanv/learning_vision3d/datasets/snet_pre_rendered_masks_all_100_new"

    # class_ids: str = "02691156,03001627,03790512"
    class_ids: str = "02858304,02924116,03790512,04468005,\
    02992529,02843684,02954340,02691156,\
    02933112,03001627,03636649,04090263,\
    04379243,04530566,02828884,02958343,\
    03211117,03691459,04256520,04401088,\
    02747177,02773838,02801938,02808440,\
    02818832,02834778,02871439,02876657,\
    02880940,02942699,02946921,03085013,\
    03046257,03207941,03261776,03325088,\
    03337140,03467517,03513137,03593526,\
    03624134,03642806,03710193,03759954,\
    03761084,03797390,03928116,03938244,\
    03948459,03991062,04004475,04074963,\
    04099429,04225987,04330267,04460130,04554684"
    name: str = "warehouse3d"
    bs_train: int = 4
    bs_val: int = 1
    bs_test: int = 1
    num_workers: int = 0


@dataclass
class GenericImgMaskData(DataConfig):
    name: str = "common_dl"
    train_split: float = 0.9
    validation_split: float = 0.1
    train_dataset_file: str = "path_to_class_img_path_maskpath_file.csv"
    val_dataset_file: str = "path_to_class_img_path_maskpath_file.csv"
    mask_path_prefix: str = ""
    rgb_path_prefix: str = ""

    class_ids: str = "class1,class2"
    max_per_class: int = 3500

    bs_train: int = 1
    bs_val: int = 1
    bs_test: int = 1


cs.store(group="data", name="default", node=DataConfig)
cs.store(group="data", name="generic_img_mask", node=GenericImgMaskData)
cs.store(group="data", name="warehouse3d", node=Warehouse3DData)


@dataclass
class RenderConfig:
    img_size: int = 128
    focal_length: int = 300
    near_plane: float = 0.1  # TODO:tune this
    far_plane: float = 2.5  # TODO:tune this
    camera_near_dist: float = 1.3  # TODO: tune this
    camera_far_dist: float = 1.7  # TODO: tune this
    cam_num: int = 5  # if -1, render on fly, dont use prerend
    num_pre_rend_masks: int = 50  # -1 corresponds to use all
    ray_num_per_cam: int = 340  # TODO: tune this
    on_ray_num_samples: int = 80  # TODO: tune this
    rgb: bool = True
    normals: bool = False
    depth: bool = False

    # No camera pose params
    softmin_temp: float = 1.0
    loss_mode: str = "softmax"  # other option is "softmax"
    use_momentum: bool = True


cs.store(group="render", name="default", node=RenderConfig)


@dataclass
class CheckpointConfig:
    name: Any = None
    version: int = 0
    epoch: Any = "last"
    pl_module: Any = None


def extract_ckpt_path(cfg):
    path = osp.join(cfg.name, "version_{}".format(cfg.version))
    if cfg.epoch == "last":
        checkpoint_path = osp.join(path, "checkpoints", "last.ckpt")
    else:
        checkpoint_path = osp.join(
            path, "checkpoints", "epoch={}.ckpt".format(cfg.epoch)
        )
    return checkpoint_path


@dataclass
class OptimizationConfig:
    val_check_interval: float = 1  # 300
    num_val_iter: int = 20
    save_freq: int = 25
    max_epochs: int = 3000
    stage_one_epochs: int = 2
    lr: float = 0.00005
    use_scheduler: bool = False
    use_pretrain: bool = False
    checkpoint_path: str = "somepath"

    use_shape_reg: bool = False


cs.store(group="optim", name="default", node=OptimizationConfig)


@dataclass
class LoggingConfig:
    log_dir: str = "job_outputs"
    name: str = "temp"


cs.store(group="logging", name="tensorboard", node=LoggingConfig)


@dataclass
class ModelConfig:
    encoder: str = ""
    decoder: str = ""
    c_dim: int = 0
    inp_dim: int = 3
    fine_tune: str = "all"  # "encoder" or "decoder" or "none"


@dataclass
class VNetConfig(ModelConfig):
    encoder: str = "resnet34_res_fc"
    decoder: str = "siren_rgb"
    c_dim: int = 2560
    inp_dim: int = 3
    fine_tune: str = "all"  # "encoder" or "decoder" or "none"


cs.store(group="model", name="default", node=ModelConfig)
cs.store(group="model", name="vnet", node=VNetConfig)


@dataclass
class ResourceConfig:
    gpus: int = 1
    num_nodes: int = 1
    num_workers: int = 0
    accelerator: Any = "ddp"  # ddp or dp or none

    # cluster specific config
    use_cluster: bool = False
    max_mem: bool = False  # if true use volta32gb for SLURM jobs.
    time: int = 60 * 36  # minutes
    partition: str = "dev"
    comment: str = "please fill this if using priority partition"

    # TOOD: later remove this
    mesh_th: float = 2.0


cs.store(group="resources", name="default", node=ResourceConfig)


@dataclass
class DistillationConfig:
    # name: str = ""
    mode: str = "point"

    num_points: int = 1000
    sample_bounds: List[float] = field(default_factory=lambda: [-0.6, 0.6])

    use_encoder_transforms: bool = False

    warehouse3d_prob: float = 0.3
    warehouse3d_ckpt_path: str = ""

    ckpts_root_dir: str = ""
    regex_exclude: str = ""
    regex_match: str = ""


cs.store(group="distillation", name="empty", node=DistillationConfig)


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)

    data: DataConfig = MISSING
    render: RenderConfig = MISSING
    logging: LoggingConfig = MISSING
    model: ModelConfig = MISSING
    optim: OptimizationConfig = MISSING
    resources: ResourceConfig = MISSING
    distillation: DistillationConfig = MISSING


cs.store(name="config", node=Config)

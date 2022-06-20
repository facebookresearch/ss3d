# Installation

The basic installation includes dependencies like pytorch, pytorch3d, pytorch-lightning etc.
  ```
  git clone https://github.com/facebookresearch/ss3d.git
  cd ss3d
  conda env create -f env.yaml
  conda activate ss3d
  ```


# Data Peperation 

For synthetic pre-training we rely from Warehouse3D models as specified in the Shpaenet3D-Core split. For generating pre-rendered images of this dataset, we recommend users to use [this](https://github.com/shubhtuls/snetRenderer) blender rendering tool for Shapenet.

The generated images should be of the following structure - 
```
    synthetic_rendered_images_root
    |-- class_a # In shapenet case, it would be synset id's
    |-- ...
    |-- ...
    |-- class_n
        |-- render_0.png # rgb rendered images
        |-- ...
        |-- render_100.png
        |-- depth_0.png # depth images used to extract mask images
        |-- ...
        |-- depht_100.png
        |-- camera_0.mat # camera intrinsics and exterinsics matrices for the rendered images.
        |-- ...
        |-- camera_100.mat
```

For category-specific training, for any of the datasets you wish to work with, please generate .csv files for training and test phases 
respectively of the format,

```
class_name rgb_image_path mask_image_path Bounding_BOX_X_min, Bounding_BOX_Y_min, Bounding_BOX_X_max, Bounding_BOX_Y_max
```

For datasets which already have bounding box cropped images of single object instaces, the following format .csv files is acceptable too for 
training and test phases,

```
class_name rgb_image_path mask_image_path
```

# Training 
To understand the config overrides more we encourage users to go through the config file located at `hydra_config/config.py`.

For synthetic pretraining,
```
python synth_pretraining.py resources.gpus=8 resources.num_nodes=4 resources.use_cluster=True \
logging.name=synthetic_pretraining optim.use_scheduler=True
```

For category-specific finetuning,
```
python cat_spec_training.py \
resources.use_cluster=True resources.gpus=8 resources.num_nodes=2 \
logging.name="<your_logging_dir_name>" \
render.cam_num=10 render.num_pre_rend_masks=10 \
data=generic_img_mask data.bs_train=4 \
data.train_dataset_file="<path_to_train_csv" \
data.val_dataset_file="<path_to_val_csv" \
data.class_ids="class_id_as_seen_in_csv" \
optim.stage_one_epochs=10 optim.max_epochs=50 optim.lr=0.00001 \
optim.use_pretrain=True \
optim.checkpoint_path="<path_to_your_synthetic_pretrained_ckpt>"
```

For learning unfified model via distillation,
```
python unified_distillation.py \
resources.use_cluster=True resources.gpus=8 resources.num_nodes=4 \
logging.name="<your_logging_dir_name>" \
render.ray_num_per_cam=10 render.cam_num=1 \
data.bs_train=14 \
optim.lr=0.00001 optim.max_epochs=75 \
optim.use_pretrain=True optim.checkpoint_path="<path_to_your_synthetic_pretrained_ckpt>" \
distillation.num_points=25000 \
distillation.mode=point \
distillation.regex_match="*49.ckpt" \
distillation.ckpts_root_dir="<root dir where all category specific model checkpoints are stored>" \
distillation.warehouse3d_ckpt_path="<path_to_your_synthetic_pretrained_ckpt>"
```
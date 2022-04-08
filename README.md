# [Pre-train, Self-train, Distill: A simple recipe for Supersizing 3D Reconstruction](https://shubhtuls.github.io/ss3d/)
Kalyan Vasudev Alwala, Abhinav Gupta, Shubham Tulsiani

[[Paper](https://arxiv.org/abs/2204.03642)] [[Project Page](https://shubhtuls.github.io/ss3d/)]

<img src="https://shubhtuls.github.io/ss3d/resources/teaser.gif" width="50%">

## Setup
Download the final distilled model from [here](https://dl.fbaipublicfiles.com/ss3d/distilled_model.torch).

Install the following pre-requisites:
* Python >=3.6
* PyTorch tested with `1.10.0` 
* TorchVision tested with `0.11.1`
* Trimesh
* pymcubes

## 3D Reconstruction Interface

Reconstruct 3D in 3 simple simple steps! Please see the [demo notebook](demo.ipynb) for a working example.

```python

# 1. Load the pre-trained checkpoint
model_3d = VNet()
model_3d.load_state_dict(torch.load("<Path to the Model>"))
model_3d.eval()


# 2. Preprocess an RGB image with associated object mask according to our model's input interface
inp_img = generate_input_img(
    img_rgb,
    img_mask,
)

# 3. Obtain 3D prediction!
out_mesh = extract_trimesh(model_3d, inp_img, "cuda")
# To save the mesh
out_mesh.export("out_mesh_pymcubes.obj")
# To visualize the mesh
out_mesh.show()
```

## Training and Evaluation
* coming soon


## Citation
If you find the project useful for your research, please consider citing:-
```
@inproceedings{vasudev2022ss3d,
  title={Pre-train, Self-train, Distill: A simple recipe for Supersizing 3D Reconstruction},
  author={Vasudev, Kalyan Alwala and  Gupta, Abhinav and Tulsiani, Shubham},
  year={2022},
  booktitle={Computer Vision and Pattern Recognition (CVPR)}
}
```

## Contributing
We welcome your pull requests! Please see [CONTRIBUTING](CONTRIBUTING.md) and [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md) for more information.

## License
ss3d is released under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for additional details. However the Sire implementation is additionally licensed under the MIT license (see [NOTICE](NOTICE) for additional details).

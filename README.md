<div align="center">

<h1>BEVBert: Multimodal Map Pre-training for <br /> Language-guided Navigation</h1>

<div>
    <a href='https://marsaki.github.io/' target='_blank'>Dong An</a>;
    <a href='https://sites.google.com/site/yuankiqi/home' target='_blank'>Yuankai Qi</a>;
    <a href='https://scholar.google.com/citations?user=a7AMvgkAAAAJ&hl=zh-CN'>Yangguang Li</a>;
    <a href='https://yanrockhuang.github.io/' target='_blank'>Yan Huang</a>;
    <a href='http://scholar.google.com/citations?user=8kzzUboAAAAJ&hl=zh-CN' target='_blank'>Liang Wang</a>;
    <a href='https://scholar.google.com/citations?user=W-FGd_UAAAAJ&hl=en' target='_blank'>Tieniu Tan</a>;
    <a href='https://amandajshao.github.io/' target='_blank'>Jing Shao</a>;
</div>

<h3><strong>Accepted to <a href='https://iccv2023.thecvf.com/' target='_blank'>ICCV 2023</a></strong></h3>

<h3 align="center">
  <a href="https://arxiv.org/pdf/2212.04385.pdf" target='_blank'>Paper</a>
</h3>
</div>

## Abstract

Large-scale pre-training has shown promising results on the vision-and-language navigation (VLN) task. However, most existing pre-training methods employ discrete panoramas to learn visual-textual associations. This requires the model to implicitly correlate incomplete, duplicate observations within the panoramas, which may impair an agent’s spatial understanding. Thus, we propose a new map-based pre-training paradigm that is spatial-aware for use in VLN. Concretely, we build a local metric map to explicitly aggregate incomplete observations and remove duplicates, while modeling navigation dependency in a global topological map. This hybrid design can balance the demand of VLN for both short-term reasoning and long-term planning. Then, based on the hybrid map, we devise a pre-training framework to learn a multimodal map representation, which enhances spatial-aware cross-modal reasoning thereby facilitating the language-guided navigation goal. Extensive experiments demonstrate the effectiveness of the map-based pre-training route for VLN, and the proposed method achieves state-ofthe-art on four VLN benchmarks (R2R, R2R-CE, RxR, REVERIE).

## Method

![](assets/method.png)

## TODOs

* [X] Release VLN (R2R, RxR, REVERIE) code.
* [ ] Release VLN-CE (R2R-CE) code.
* [X] Data preprocessing code.
* [ ] Release checkpoints and preprocessed datasets.

## Setup

### Installation

1. Create a virtual environment. We develop this project with Python 3.6.

   ```bash
   conda env create -f environment.yaml
   ```
2. Install the latest version of [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator), including the Matterport3D RGBD datasets (for step 6).
3. Download the Matterport3D scene meshes

```bash
# run with python 2.7
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
# Extract to: ./data/scene_datasets/mp3d/{scene}/{scene}.glb
```

`download_mp.py` must be obtained from the Matterport3D [project webpage](https://niessner.github.io/Matterport/).

Follow the [Habitat Installation Guide](https://github.com/facebookresearch/habitat-lab#installation) to install [`habitat-sim`](https://github.com/facebookresearch/habitat-sim) and [`habitat-lab`](https://github.com/facebookresearch/habitat-lab). We use version [`v0.1.7`](https://github.com/facebookresearch/habitat-lab/releases/tag/v0.1.7) in our experiments. In brief:

4. Install `habitat-sim` for a machine with multiple GPUs or without an attached display (i.e. a cluster):

   ```bash
   conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless
   ```
5. Clone `habitat-lab` from the github repository and install. The command below will install the core of Habitat Lab as well as the habitat_baselines.

   ```bash
   git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-lab.git
   cd habitat-lab
   python setup.py develop --all # install habitat and habitat_baselines
   ```

6. Grid feature preprocessing for metric mapping (~100G). 

   ```bash
   python precompute_features/grid_mp3d_clip.py       # R2R, RxR
   python precompute_features/grid_mp3d_imagenet.py   # REVERIE
   python precompute_features/grid_habitat_clip.py    # R2R-CE
   python precompute_features/grid_depth.py           # grid depth
   python precompute_features/grid_sem.py             # grid semantic for pre-training
   ```

## Running

Pre-training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/pt_r2r.bash 2333  # R2R
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/pt_rxr.bash 2333  # RxR
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/pt_rvr.bash 2333  # REVERIE
```

Fine-tuning and Testing

```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/ft_r2r.bash 2333  # R2R
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/ft_rxr.bash 2333  # RxR
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/ft_rvr.bash 2333  # REVERIE
```

# Contact Information

* dong DOT an AT cripac DOT ia DOT ac DOT cn, [Dong An](https://marsaki.github.io/)
* yhuang AT nlpr DOT ia DOT ac DOT cn, [Yan Huang](https://yanrockhuang.github.io/)

# Acknowledge

Our implementations are partially inspired by [DUET](https://github.com/cshizhe/VLN-DUET), [S-MapNet](https://github.com/vincentcartillier/Semantic-MapNet) and [ETPNav](https://github.com/MarSaKi/ETPNav).

Thank them for open sourcing their great works!

# Citation

If you find this repository is useful, please consider citing our paper:

```
@article{an2022bevbert,
  title={BEVBert: Multimodal Map Pre-training for Language-guided Navigation},
  author={An, Dong and Qi, Yuankai and Li, Yangguang and Huang, Yan and Wang, Liang and Tan, Tieniu and Shao, Jing},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

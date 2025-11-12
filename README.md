# Unofficial Implementation of VistaDream: Sampling multiview consistent images for single-view scene reconstruction

**VistaDream** is a novel framework for reconstructing 3D scenes from single-view images using Flux-based diffusion models. This implementation combines image outpainting, depth estimation, and 3D Gaussian splatting for high-quality 3D scene generation, with integrated visualization using [Rerun](https://rerun.io/).

Uses [Rerun](https://rerun.io/) for 3D visualization, [Gradio](https://www.gradio.app) for interactive UI, [Flux](https://github.com/black-forest-labs/flux) for diffusion-based outpainting, and [Pixi](https://pixi.sh/latest/) for easy installation.

<p align="center">
  <a title="Paper" href="https://vistadream-project-page.github.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
    <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
  </a>
  <a title="Rerun" href="https://rerun.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
    <img src="https://img.shields.io/badge/Rerun-0.24.0-blue.svg?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzQ0MV8xMTAzOCkiPgo8cmVjdCB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHJ4PSI4IiBmaWxsPSJibGFjayIvPgo8cGF0aCBkPSJNMy41OTcwMSA1Ljg5NTM0TDkuNTQyOTEgMi41MjM1OUw4Ljg3ODg2IDIuMTQ3MDVMMi45MzMgNS41MTg3NUwyLjkzMjk1IDExLjI5TDMuNTk2NDIgMTEuNjY2MkwzLjU5NzAxIDUuODk1MzRaTTUuMDExMjkgNi42OTc1NEw5LjU0NTc1IDQuMTI2MDlMOS41NDU4NCA0Ljk3NzA3TDUuNzYxNDMgNy4xMjI5OVYxMi44OTM4SDcuMDg5MzZMNi40MjU1MSAxMi41MTczVjExLjY2Nkw4LjU5MDY4IDEyLjg5MzhIOS45MTc5NUw2LjQyNTQxIDEwLjkxMzNWMTAuMDYyMUwxMS40MTkyIDEyLjg5MzhIMTIuNzQ2M0wxMC41ODQ5IDExLjY2ODJMMTMuMDM4MyAxMC4yNzY3VjQuNTA1NTlMMTIuMzc0OCA0LjEyOTQ0TDEyLjM3NDMgOS45MDAyOEw5LjkyMDkyIDExLjI5MTVMOS4xNzA0IDEwLjg2NTlMMTEuNjI0IDkuNDc0NTRWMy43MDM2OUwxMC45NjAyIDMuMzI3MjRMMTAuOTYwMSA5LjA5ODA2TDguNTA2MyAxMC40ODk0TDcuNzU2MDEgMTAuMDY0TDEwLjIwOTggOC42NzI1MlYyLjk5NjU2TDQuMzQ3MjMgNi4zMjEwOUw0LjM0NzE3IDEyLjA5Mkw1LjAxMDk0IDEyLjQ2ODNMNS4wMTEyOSA2LjY5NzU0Wk05LjU0NTc5IDUuNzMzNDFMOS41NDU4NCA4LjI5MjA2TDcuMDg4ODYgOS42ODU2NEw2LjQyNTQxIDkuMzA5NDJWNy41MDM0QzYuNzkwMzIgNy4yOTY0OSA5LjU0NTg4IDUuNzI3MTQgOS41NDU3OSA1LjczMzQxWiIgZmlsbD0id2hpdGUiLz4KPC9nPgo8ZGVmcz4KPGNsaXBQYXRoIGlkPSJjbGlwMF80NDFfMTEwMzgiPgo8cmVjdCB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIGZpbGw9IndoaXRlIi8+CjwvY2xpcFBhdGg+CjwvZGVmcz4KPC9zdmc+Cg==">
  </a>
  <a title="Github" href="https://github.com/rerun-io/vistadream" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
    <img src="https://img.shields.io/github/stars/rerun-io/vistadream?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
  </a>
</p>

<p align="center">
  <img src="media/readme-office-recon-small.gif" alt="VistaDream 3D scene reconstruction" width="720" />
</p>

## Overview

VistaDream addresses the challenge of 3D scene reconstruction from a single image through a novel two-stage pipeline:

1. **Coarse 3D Scaffold Construction**: Creates a global scene structure by outpainting image boundaries and estimating depth maps
2. **Multi-view Consistency Sampling (MCS)**: Uses iterative diffusion-based RGB-D inpainting with multi-view consistency constraints to generate high-quality novel views

The framework integrates multiple state-of-the-art models:
- **Flux diffusion models** for high-quality image outpainting and inpainting
- **3D Gaussian Splatting** for efficient 3D scene representation
- **Rerun** for real-time 3D visualization and debugging

## Installation

### Prerequisites
- **Linux only** with **NVIDIA GPU** (tested with CUDA 12.9)
- [Pixi](https://pixi.sh/latest/#installation) package manager

**NOTE:** You may need to change the **CUDA version** and **CUDA compute capability** in `pyproject.toml` (look for `cuda-version` and `TORCH_CUDA_ARCH_LIST`, respectively). You can find your CUDA version by running `nvidia-smi` or `nvcc --version` and your CUDA compute capability by running `nvidia-smi --query-gpu=compute_cap --format=csv` or check [Nvidia webiste](https://developer.nvidia.com/cuda-gpus).

### Using Pixi

```bash
git clone https://github.com/rerun-io/vistadream.git
cd vistadream
pixi run example
```

This will automatically download the required models and run the example with the included office image.

## Usage

For the commands below you can add the `--help` flag to see more options, for example `pixi run python tools/run_single_img.py --help`.

### Single Image Processing
Process a single image with depth estimation and basic 3D reconstruction:

```bash
pixi run python tools/run_single_img.py --image-path data/office/IMG_4029.jpg
```

### Flux Outpainting Only
Run just the outpainting component with Rerun visualization:

```bash
pixi run python tools/run_flux_outpainting.py --image-path data/office/IMG_4029.jpg --expansion-percent 0.2
```

### Multi-Image Pose & Depth Pipeline (VGGT + MoGe)
Estimate camera intrinsics/extrinsics, per-image depth, confidence masks, and fuse them into an (optionally downsampled) colored point cloud from a directory of images. Results stream live to a Rerun viewer.

```bash
pixi run python tools/run_multi_img.py --image-dir /path/to/image_folder
```

Connect to an already running Rerun viewer (instead of spawning a new one):

```bash
pixi run python tools/run_multi_img.py --rr-config.connect --image-dir /path/to/image_folder
```

Notes:
- Supported image extensions: .png, .jpg, .jpeg
- Automatically orients & recenters camera poses ("up" orientation heuristic) and logs a consolidated point cloud plus per‑view RGB, depth, filtered depth, MoGe depth, and confidence.
- Uses VGGT (multiview geometry transformer) for joint pose & depth, robust depth confidence filtering, MoGe for refined monocular depth, and voxel downsampling to target a manageable point count.

### Gradio Web Interface
Launch an interactive web interface for experimenting with the models:

```bash
pixi run python tools/gradio_app.py
```

## Key Features

- **Single Image to 3D**: Complete pipeline from single image to navigable 3D scene
- **Multi-Image Geometry**: Batch multi-view camera & depth estimation with fused colored point cloud export
- **Memory Efficient**: Model offloading support for GPU memory management
- **Real-time Visualization**: Integrated Rerun viewer for 3D scene inspection
- **Training-free**: No fine-tuning required for existing diffusion models
- **High Quality**: Multi-view consistency sampling ensures coherent 3D reconstruction

## Project Structure

```
├── src/vistadream/
│   ├── api/                 # High-level pipeline APIs
│   │   ├── flux_outpainting.py    # Outpainting-only pipeline
│   │   ├── multi_image_pipeline.py # Multi-image pose & depth fusion (VGGT + MoGe)
│   │   └── vistadream_pipeline.py # Full 3D reconstruction pipeline
│   ├── flux/                # Flux diffusion model integration
│   │   ├── cli_*.py         # Command-line interfaces
│   │   ├── model.py         # Flux transformer architecture
│   │   ├── sampling.py      # Diffusion sampling logic
│   │   └── util.py          # Model loading and configuration
│   └── ops/                 # Core operations
│       ├── flux.py          # Flux model wrappers
│       ├── gs/              # Gaussian splatting implementation
│       ├── trajs/           # Camera trajectory generation
│       └── visual_check.py  # 3D scene validation tools
└── tools/                   # Standalone applications
    ├── gradio_app.py        # Web interface
    ├── run_flux_outpainting.py
    ├── run_vistadream.py    # Main 3D pipeline
    └── run_single_img.py    # Single image processing
```

## Model Checkpoints

Models are automatically downloaded from Hugging Face on first run. Manual download:

```bash
pixi run huggingface-cli download pablovela5620/vistadream --local-dir ckpt/
```

Expected structure:
```
ckpt/
├── flux_fill/
│   ├── flux1-fill-dev.safetensors
│   └── ae.safetensors
├── vec.pt
├── txt.pt
└── txt_256.pt
```

## Citation

Thanks to the original authors! If you use VistaDream in your research, please cite:

[Original Repo](https://github.com/WHU-USI3DV/VistaDream)
```bibtex
@inproceedings{wang2025vistadream,
  title={VistaDream: Sampling multiview consistent images for single-view scene reconstruction},
  author={Wang, Haiping and Liu, Yuan and Liu, Ziwei and Wang, Wenping and Dong, Zhen and Yang, Bisheng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```

## Acknowledgements

This project builds upon several outstanding works:

- **Flux** - [Black Forest Labs](https://github.com/black-forest-labs/flux) for the diffusion model foundation
- **3D Gaussian Splatting** - [Inria](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) for efficient 3D representation
- **Rerun** - [Rerun.io](https://rerun.io/) for 3D visualization framework
- **GSplat** - [Nerfstudio](https://github.com/nerfstudio-project/gsplat) for Gaussian splatting implementation
- **MoGe** - [Microsoft Research](https://github.com/microsoft/MoGe) for monocular geometry estimation

## Related Work

- **[ASUKA](https://github.com/Yikai-Wang/asuka-misato)** - Enhanced image inpainting for mitigating unwanted object insertion
- **[MoGe](https://wangrc.site/MoGePage/)** - Accurate monocular geometry estimation for open-domain images

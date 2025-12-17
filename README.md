# napari-voxtell

[![GitHub](https://img.shields.io/badge/GitHub-VoxTell-181717?logo=github&logoColor=white)](https://github.com/MIC-DKFZ/VoxTell)&#160;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-VoxTell-yellow)](https://huggingface.co/MIC-DKFZ/VoxTell)&#160;
[![arXiv](https://img.shields.io/badge/arXiv-2511.11450-B31B1B.svg)](https://arxiv.org/abs/2511.11450)
[![napari](https://img.shields.io/badge/napari-plugin-80d1ff)](https://github.com/MIC-DKFZ/napari-voxtell)

**A napari plugin for text-promptable 3D medical image segmentation.**

## Description

**napari-voxtell** integrates [VoxTell](https://github.com/MIC-DKFZ/VoxTell), a 3D vision-language segmentation model, into the napari ecosystem. This plugin enables text-based prompting for volumetric medical image segmentation, offering an alternative to traditional interaction methods such as bounding boxes, point clicks, or manual brush strokes, used e.g. in our [nnInteractive plugin](https://github.com/MIC-DKFZ/napari-nninteractive).

VoxTell accepts free-form text descriptions (e.g., "liver", "aortic arch", "brain tumor") to generate 3D segmentation masks. As an experimental research tool, napari-voxtell is designed to facilitate exploration and prototyping in medical image analysis workflows rather than production clinical use.

**Note:** VoxTell is an ongoing research project and may produce variable results depending on anatomical region, imaging modality, and prompt specificity. Users should validate outputs carefully and not rely on this tool for clinical decision-making without expert review.

## Features

- 🗣️ **Text-based prompting**: Segment anatomical structures and pathologies using natural language descriptions.
- 🧠 **Multi-modality support**: Compatible with CT, MRI, and PET volumetric data.
- 🔌 **Seamless napari integration**: Select image layers and visualize results directly within the napari viewer.
- ⚙️ **Flexible model loading**: Switch between model versions or load custom checkpoints for experimentation.

## Important: Image Orientation and Spacing

- ⚠️ **Image Orientation (Critical)**: For correct anatomical localization (e.g., distinguishing left from right), images **must be in RAS orientation**. VoxTell was trained on data reoriented using [this specific reader](https://github.com/MIC-DKFZ/nnUNet/blob/86606c53ef9f556d6f024a304b52a48378453641/nnunetv2/imageio/nibabel_reader_writer.py#L101). While this plugin attempts to handle reorientation under the hood, mismatches can be a source of error. An easy way to test for this is if a simple prompt like "liver" fails and segments e.g. parts of the spleen instead.

- **Image Spacing**: The model does not resample images to a standardized spacing for faster inference. Performance may degrade on images with very uncommon voxel spacings (e.g., super high-resolution brain MRI). In such cases, consider resampling the image to a more typical clinical spacing (e.g., 1.5×1.5×1.5 mm³) before segmentation.


## Installation

### 1. Create a virtual environment

VoxTell supports Python 3.10+ and works with Conda, pip, or any other virtual environment. Here's an example using Conda:

```
conda create -n voxtell python=3.12
conda activate voxtell
```

### 2. Install PyTorch

> [!WARNING]
> **Temporary Compatibility Warning**  
> There is a known issue with **PyTorch 2.9.0** causing **OOM errors during inference** (related to 3D convolutions — see the PyTorch issue [here](https://github.com/pytorch/pytorch/issues/166122)).  
> **Until this is resolved, please use PyTorch 2.8.0 or earlier.**

Install PyTorch compatible with your CUDA version. For example, for Ubuntu with a modern Nvidia GPU:

```
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
```

*For other configurations (Mac, CPU, different CUDA versions), please refer to the [PyTorch Get Started](https://pytorch.org/get-started/previous-versions/) page.*

### 3. Install napari-voxtell

You can install the plugin directly from the repository:

```
git clone https://github.com/MIC-DKFZ/napari-voxtell
cd napari-voxtell
pip install -e .
```

> **Note:** Model weights are automatically downloaded from Hugging Face on first use. This may take a few minutes depending on your internet connection.

## Getting Started

You can launch the plugin in three ways.

**Note:** If asked which plugin to use for opening `.nii.gz` files, we recommend selecting `napari-nifti`.

**Option A: Start napari and activate manually**
```
napari
```
Then go to **Plugins** > **napari-voxtell**.

**Option B: Start napari with the widget open**
```
napari -w napari-voxtell
```

**Option C: Open an image directly with the widget**
```
napari path/to/your/image.nii.gz -w napari-voxtell
```

## Usage

1. **Initialize the Model**: 
   - Open the VoxTell widget.
   - Select your model version from the dropdown (or paste a local custom model path).
   - Click **Initialize**. This downloads model weights on first use and takes some time while the model loads.
2. **Select Input**:
   - Choose the target image layer from the dropdown menu.
3. **Prompt**:
   - Enter a text description of the anatomical structure or pathology of interest (e.g., "right kidney", "lung lesion", "brainstem").
4. **Segment**:
   - Click **Submit**.
   - The resulting segmentation will appear as a new Labels layer.


<p align="center">
    <img src="imgs/gui.png" alt="napari-voxtell GUI">
</p>

Please carefully review all segmentation outputs. Model performance varies with anatomical complexity, imaging quality, spacing, and prompt clarity. This tool is intended for research exploration, not validated clinical workflows.


## Citation

If you use `napari-voxtell` in your research, please cite our paper:

```
@misc{rokuss2025voxtell,
      title={VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation}, 
      author={Maximilian Rokuss and Moritz Langenberg and Yannick Kirchhoff and Fabian Isensee and Benjamin Hamm and Constantin Ulrich and Sebastian Regnery and Lukas Bauer and Efthimios Katsigiannopulos and Tobias Norajitra and Klaus Maier-Hein},
      year={2025},
      eprint={2511.11450},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.11450}, 
}
```

## License

This repository is licensed under the **Apache-2.0 License**.

**Important:** The default model checkpoints downloaded by this plugin are licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC-BY-NC-SA 4.0)**. Please review the [Hugging Face Model Card](https://huggingface.co/mrokuss/VoxTell) for details regarding model usage and limitations.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for bugs and feature requests.

## 📬 Contact

Special shoutout to [Benjamin Hamm](https://github.com/hammb) who created the first version of this plugin. For questions, issues, or collaborations, please contact:

📧 [maximilian.rokuss@dkfz-heidelberg.de](mailto:maximilian.rokuss@dkfz-heidelberg.de) / [benjamin.hamm@dkfz-heidelberg.de](mailto:benjamin.hamm@dkfz-heidelberg.de)

______________________________________________________________________

## Acknowledgments

<p align="left">
  <img src="imgs/Logos/DKFZ_Logo.png" width="500">
</p>

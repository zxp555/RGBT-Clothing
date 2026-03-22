# Physical Adversarial Clothing Evades Visible-Thermal Detectors via Non-Overlapping RGB-T Pattern

This is the official implementation repository for the paper "Physical Adversarial Clothing Evades Visible-Thermal Detectors via Non-Overlapping RGB-T Pattern" in CVPR 2026.

## Installation

Create a conda virtual environment and activate it:

```bash
conda create -y -n attack3d python=3.9
conda activate attack3d
```

Install pytorch and its basic depencies:

```bash
conda install -y -c pytorch \
    pytorch==1.12.1 \
    torchvision==0.13.1 \
    cudatoolkit=11.3 \
    numpy==1.23.5
```

Then install other dependencies:

```bash
pip install -r requirements.txt
```

After that, install mm series:

```bash
python -m pip install mmengine==0.10.5
python -m mim install "mmcv==2.0.0rc4"
python -m pip install --no-build-isolation -e ./model/mmdetection
```

If you meet any issues, please consider compiling all of them from source following their [document](https://mmdetection.readthedocs.io/) as an alternative.

Finally, compile detectron2 from source code in `./model/eccv22` following its [instructions](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)

We used CUDA-11.3 for all the experiments.

## Run Attack

Here is an example to run attack for the earlyfusion RGB-T detector.

First, find the initial weights from [this link](https://drive.google.com/drive/folders/1Uy6ft7kWz_so-h6BHY1QcokdgFNrJQXj?usp=sharing) and put it into `assets/detection_ckpt/`.

Then prepare the dataset. We used [this dataset](https://github.com/CalayZhou/Multispectral-Pedestrian-Detection-Resource/issues/6) for our experiment. Please download this dataset to your local paths and setup the directories into `configs/default.json` accordingly. Note the paired RGB and thermal images should be with same filenames but in two directories.

After that, you can launch adversarial attack to the model on the dataset.

```bash
python attack.py --config configs/early.json
```

We also recommend you try enhancing some low-contrast RGB images in the original FLIR dataset using `util/enhance.py` before training.

## Acknowledgement

The code is built on top of [yolov9](https://github.com/ultralytics/yolov), [mmdet](https://github.com/open-mmlab/mmdetection), [detectron2](https://github.com/facebookresearch/detectron2) and [proben](https://github.com/Jamie725/Multimodal-Object-Detection-via-Probabilistic-Ensembling). We thank all the authors for their great work.

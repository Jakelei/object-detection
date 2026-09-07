# SA-DETR

**SA-DETR: A Detection Transformer with Structure-Aware Enhancement for Ocean Eddy Small Object Detection**

This repository contains the official implementation of **SA-DETR**, a structure-aware end-to-end detector for mesoscale ocean eddies in sea level anomaly (SLA) fields.

Unlike objects with clear visual boundaries, small eddies are represented by weak, spatially continuous responses that can be difficult to distinguish from the surrounding background. SA-DETR builds on Relation-DETR and introduces two task-oriented components:

- **HSAE (High-Frequency Structure-Aware Enhancement)** strengthens subtle, position-related structural variations before transformer encoding.
- **TRRO (Target-Region Representation Optimization)** consolidates the anomalous core and its surrounding responses within the encoder to form a more complete target-region representation.

The model retains end-to-end set prediction with Hungarian assignment and is trained with the same classification, L1, and GIoU objectives as its Relation-DETR baseline.

> [!IMPORTANT]
> This repository is an initial public preview. The core implementations listed below are intentionally not included in this release and will be provided in a later update:
>
> - `models/bricks/hsae.py`
> - `models/bricks/trro.py`
> - `models/necks/channel_mapper_hsae.py`
>
> The remaining source tree, configuration, environment, and usage instructions are provided for inspection. Full SA-DETR training and evaluation require these three modules.

## Repository structure

```text
SA-DETR/
├── configs/
│   ├── sa_detr/
│   │   └── sa_detr_resnet50_800_1333.py
│   └── train_config.py
├── datasets/
├── models/
│   ├── backbones/
│   ├── bricks/
│   ├── detectors/
│   │   └── sa_detr.py
│   ├── matcher/
│   └── necks/
├── optimizer/
├── transforms/
├── util/
├── main.py
├── test.py
└── requirements.txt
```

Datasets, checkpoints, logs, prediction files, and visualization results are not distributed in this repository.

## Environment

The reference environment follows Salience-DETR and Relation-DETR:

```bash
git clone https://github.com/Jakelei/object-detection.git SA-DETR
cd SA-DETR

conda create -n sdetr python=3.8 -y
conda activate sdetr

conda install pytorch==1.12.1 torchvision==0.13.1 \
  torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
```

The HSAE implementation additionally uses `einops` and `torch-dct`; both are included in `requirements.txt`.

Verify the main framework versions after installation:

```bash
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

## Dataset preparation

SA-DETR uses COCO-format annotations. Arrange each dataset as follows:

```text
/path/to/dataset/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

Set the dataset location through `SA_DETR_DATA_ROOT`:

```bash
export SA_DETR_DATA_ROOT=/path/to/dataset
```

For a dataset with a different number of categories, update `num_classes` in `configs/sa_detr/sa_detr_resnet50_800_1333.py`.

## Training

The commands in this section document the intended training interface and become executable for SA-DETR when the three core modules identified above are available.

Activate the environment, set the dataset root, and launch training from the repository root.

Single GPU:

```bash
conda activate sdetr
export SA_DETR_DATA_ROOT=/path/to/dataset
CUDA_VISIBLE_DEVICES=0 accelerate launch main.py \
  --config-file configs/train_config.py
```

Multiple GPUs:

```bash
conda activate sdetr
export SA_DETR_DATA_ROOT=/path/to/dataset
CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py \
  --config-file configs/train_config.py
```

Training parameters such as epochs, batch size, learning rate, output directory, and model configuration are defined in `configs/train_config.py`.

## Resume training or fine-tune

Set `SA_DETR_RESUME` before running the same training command.

- A previous **training directory** restores the model, optimizer, scheduler, and epoch state from its latest checkpoint.
- A **`.pth` checkpoint** loads model weights for fine-tuning but does not restore the complete optimizer and scheduler state.

Resume a complete run:

```bash
export SA_DETR_DATA_ROOT=/path/to/dataset
export SA_DETR_RESUME=/path/to/previous/training_directory
CUDA_VISIBLE_DEVICES=0 accelerate launch main.py \
  --config-file configs/train_config.py
```

Fine-tune from model weights:

```bash
export SA_DETR_DATA_ROOT=/path/to/dataset
export SA_DETR_RESUME=/path/to/checkpoint.pth
CUDA_VISIBLE_DEVICES=0 accelerate launch main.py \
  --config-file configs/train_config.py
```

## Evaluation and test

The same release-status note applies to SA-DETR evaluation.

Evaluate a checkpoint on the validation set:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch test.py \
  --coco-path /path/to/dataset \
  --model-config configs/sa_detr/sa_detr_resnet50_800_1333.py \
  --checkpoint /path/to/checkpoint.pth
```

Save predictions as a COCO-format JSON file:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch test.py \
  --coco-path /path/to/dataset \
  --model-config configs/sa_detr/sa_detr_resnet50_800_1333.py \
  --checkpoint /path/to/checkpoint.pth \
  --result predictions.json
```

Visualize predictions:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch test.py \
  --coco-path /path/to/dataset \
  --model-config configs/sa_detr/sa_detr_resnet50_800_1333.py \
  --checkpoint /path/to/checkpoint.pth \
  --show-dir visualization/
```

Evaluate an existing prediction JSON on CPU:

```bash
accelerate launch test.py \
  --coco-path /path/to/dataset \
  --result /path/to/predictions.json
```

## Pretrained models

Pretrained checkpoints and training logs are not included in this initial preview. Download links will be added in a future release.

## Citation

The BibTeX entry will be added after the paper record becomes publicly available.

## License & Acknowledgment

This repository retains the Apache License 2.0 used by the Relation-DETR codebase. See [LICENSE](LICENSE) for details.

We are very grateful for these excellent works: [Relation-DETR](https://github.com/xiuqhou/Relation-DETR). Please follow their license for usage and redistribution. Thanks for the awesome works.

## Contact

Feel free to contact me if there is any question. ([Lei Chen: chenlei9958@stu.ouc.edu.cn](mailto:chenlei9958@stu.ouc.edu.cn), [Lei Huang: huangl@ouc.edu.cn](mailto:huangl@ouc.edu.cn))

# SA-DETR: A Detection Transformer with Structure-Aware Enhancement for Ocean Eddy Object Detection

This is the implementation of paper: SA-DETR: A Detection Transformer with Structure-Aware Enhancement for Ocean Eddy Object Detection.

**Note:** The source code is currently incomplete and will be fully released once the manuscript is accepted by the journal.

## Datasets

Experiments are conducted on four datasets: **Area-A**, **Area-B**, **DOTA-v1.0**, and **VisDrone2019**.

| # | Dataset | Download |
|:--:|---|:---:|
| 1 | Area-A | [Link](https://github.com/huanglab-research/Mesoscale-Eddy-Dataset) |
| 2 | Area-B | [Link](https://github.com/huanglab-research/Mesoscale-Eddy-Dataset) |
| 3 | DOTA-v1.0 | [Link](https://captain-whu.github.io/DOTA/dataset.html) |
| 4 | VisDrone2019 | [Link](https://github.com/VisDrone/VisDrone-Dataset) |

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
│   │   ├── hsae.py
│   │   ├── trro.py
│   │   └── sa_transformer.py
│   ├── detectors/
│   │   └── sa_detr.py
│   ├── matcher/
│   └── necks/
│       └── channel_mapper.py
├── optimizer/
├── transforms/
├── util/
├── main.py
├── test.py
└── requirements.txt
```

## Environment

```bash
git clone https://github.com/Jakelei/object-detection.git SA-DETR
cd SA-DETR

conda create -n sdetr python=3.8 -y
conda activate sdetr

conda install pytorch==1.12.1 torchvision==0.13.1 \
  torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
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

## Evaluation and test

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

## License & Acknowledgment

We are very grateful for these excellent works: [Relation-DETR](https://github.com/xiuqhou/Relation-DETR), [DINO](https://github.com/IDEA-Research/DINO), and [HS-FPN](https://github.com/ShiZican/HS-FPN). Please follow their respective licenses for usage and redistribution. Thanks for their awesome works.

## Contact

Feel free to contact me if there is any question. ([Lei Chen: chenlei9958@stu.ouc.edu.cn](mailto:chenlei9958@stu.ouc.edu.cn), [Lei Huang: huangl@ouc.edu.cn](mailto:huangl@ouc.edu.cn))

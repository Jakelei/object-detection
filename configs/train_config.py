"""Default SA-DETR training configuration.

Set ``SA_DETR_DATA_ROOT`` to a COCO-format dataset root containing
``train2017``, ``val2017`` and ``annotations`` before starting training.
"""

import os

from torch import optim

from datasets.coco import CocoDetection
from optimizer import param_dict
from transforms import presets


num_epochs = 24
batch_size = 2
num_workers = 2
pin_memory = True
print_freq = 100
starting_epoch = 0
max_norm = 0.1

output_dir = None
find_unused_parameters = False
resume_from_checkpoint = os.environ.get("SA_DETR_RESUME")

data_root = os.environ.get("SA_DETR_DATA_ROOT", "/path/to/coco")
train_dataset = CocoDetection(
    img_folder=os.path.join(data_root, "train2017"),
    ann_file=os.path.join(data_root, "annotations", "instances_train2017.json"),
    transforms=presets.detr,
    train=True,
)
test_dataset = CocoDetection(
    img_folder=os.path.join(data_root, "val2017"),
    ann_file=os.path.join(data_root, "annotations", "instances_val2017.json"),
    transforms=None,
)

model_path = "configs/sa_detr/sa_detr_resnet50_800_1333.py"

learning_rate = 1e-4
optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[20], gamma=0.1)
param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)

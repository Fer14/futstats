import os

import torch
import typer
from super_gradients.training import Trainer, models
from super_gradients.training.dataloaders.dataloaders import (
    coco2017_train_yolo_nas,
    coco2017_val_yolo_nas,
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val,
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import (
    DetectionMetrics,
    DetectionMetrics_050,
    DetectionMetrics_050_095,
)
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


CLASSES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
]

NUM_CLASES = len(CLASSES)
EPOCHS = 750


def main(data_format: str = "coco", train: bool = True, test: bool = False):
    print(f"Selected data format: {data_format}")
    if data_format == "coco":
        MODEL_ARCH = "yolo_nas_s"
        CHECKPOINT_DIR = "./checkpoints"

        EXPERIMENT = "FIELD_KEYPOINTS_COCO_DATA_2"
        LOCATION = "/home/fer/Escritorio/futstatistics/datasets/field/keypoints3/"
        BATCH_SIZE = 32
        INPUT_DIM = [
            640,
            640,
        ]  # you have to make sure the img size can be divided by 32. That's the rule. :)

        train_dataset_params = {
            "data_dir": LOCATION,
            "subdir": "train",
            "json_file": "train_annotations.coco.json",
            "input_dim": INPUT_DIM,
            "ignore_empty_annotations": False,
        }

        train_data = coco2017_train_yolo_nas(
            dataset_params=train_dataset_params,
            dataloader_params={"batch_size": BATCH_SIZE, "num_workers": 2},
        )

        test_dataset_params = {
            "data_dir": LOCATION,
            "subdir": "valid",
            "json_file": "valid_annotations.coco.json",
            "input_dim": INPUT_DIM,
            "ignore_empty_annotations": False,
        }

        test_data = coco2017_val_yolo_nas(
            dataset_params=test_dataset_params,
            dataloader_params={"batch_size": BATCH_SIZE, "num_workers": 2},
        )

    train_params = {
        "silent_mode": False,
        "average_best_models": True,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 6e-5,  # cambiar a uno mas bajo para que no overfitee
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": EPOCHS,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=len(CLASSES),
            reg_max=16,
            # classification_loss_weight=4.0,
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=len(CLASSES),
                normalize_targets=True,
                # calc_best_score_thresholds=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7,
                ),
            ),
            DetectionMetrics_050_095(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=len(CLASSES),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7,
                ),
            ),
        ],
        "metric_to_watch": "mAP@0.50:0.95",
    }

    model = models.get(
        MODEL_ARCH, pretrained_weights="coco", num_classes=NUM_CLASES
    ).to(DEVICE)

    # field_model = models.get(
    #     "yolo_nas_s",
    #     num_classes=NUM_CLASES,
    #     checkpoint_path="./checkpoints",
    # )

    trainer = Trainer(experiment_name=EXPERIMENT, ckpt_root_dir=CHECKPOINT_DIR)

    if train:
        trainer.train(
            model=model,
            training_params=train_params,
            train_loader=train_data,
            valid_loader=test_data,
        )


if __name__ == "__main__":
    typer.run(main)

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


# KEYPOINTS
MODEL_ARCH = "yolo_nas_m"
BATCH_SIZE = 32
CHECKPOINT_DIR = "./checkpoints"
LOCATION = "../datasets/dataset6_keypoints/dataset6_keypoints_coco"
CLASSES = [
    "0",
    "1",
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
    "2",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

NUM_CLASES = len(CLASSES)
EXPERIMENT = "FIELD_KEYPOINTS"
EPOCHS = 200


# FIELD LANDMARSK
MODEL_ARCH = "yolo_nas_l"
BATCH_SIZE = 8
CHECKPOINT_DIR = "./checkpoints"
LOCATION = "../../../datasets/dataset5_field/"
CLASSES = [
    "1",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "c",
]

NUM_CLASES = len(CLASSES)
EXPERIMENT = "FIELD_LANDMARKS"
EPOCHS = 100


def main(data_format: str = "yolo", train: bool = True, test: bool = True):

    if data_format == "coco":
        train_dataset_params = {
            "data_dir": LOCATION,
            "subdir": "images/train",
            "json_file": "train_corrected_annotations.coco.json",
            "input_dim": [320, 320],
            # "class_inclusion_list": class_inclusion_list,
            "ignore_empty_annotations": False,
        }

        train_data = coco2017_train_yolo_nas(
            dataset_params=train_dataset_params,
            dataloader_params={"batch_size": BATCH_SIZE, "num_workers": 2},
        )

        test_dataset_params = {
            "data_dir": LOCATION,
            "subdir": "images/test",
            "json_file": "test_corrected_annotations.coco.json",
            "input_dim": [320, 320],
            # "class_inclusion_list": class_inclusion_list,
            "ignore_empty_annotations": False,
        }

        test_data = coco2017_val_yolo_nas(
            dataset_params=test_dataset_params,
            dataloader_params={"batch_size": BATCH_SIZE, "num_workers": 2},
        )

        val_dataset_params = {
            "data_dir": LOCATION,
            "subdir": "images/val",
            "json_file": "val_corrected_annotations.coco.json",
            "input_dim": [320, 320],
            # "class_inclusion_list": class_inclusion_list,
            "ignore_empty_annotations": False,
        }

        val_data = coco2017_val_yolo_nas(
            dataset_params=val_dataset_params,
            dataloader_params={"batch_size": BATCH_SIZE, "num_workers": 2},
        )

    if data_format == "yolo":

        dataset_params = {
            "data_dir": LOCATION,
            "train_images_dir": "train/images",
            "train_labels_dir": "train/labels",
            "val_images_dir": "valid/images",
            "val_labels_dir": "valid/labels",
            "test_images_dir": "test/images",
            "test_labels_dir": "test/labels",
            "classes": CLASSES,
        }

        train_data = coco_detection_yolo_format_train(
            dataset_params={
                "data_dir": dataset_params["data_dir"],
                "images_dir": dataset_params["train_images_dir"],
                "labels_dir": dataset_params["train_labels_dir"],
                "classes": dataset_params["classes"],
            },
            dataloader_params={"batch_size": BATCH_SIZE, "num_workers": 2},
        )

        val_data = coco_detection_yolo_format_val(
            dataset_params={
                "data_dir": dataset_params["data_dir"],
                "images_dir": dataset_params["val_images_dir"],
                "labels_dir": dataset_params["val_labels_dir"],
                "classes": dataset_params["classes"],
            },
            dataloader_params={"batch_size": BATCH_SIZE, "num_workers": 2},
        )

        test_data = coco_detection_yolo_format_val(
            dataset_params={
                "data_dir": dataset_params["data_dir"],
                "images_dir": dataset_params["test_images_dir"],
                "labels_dir": dataset_params["test_labels_dir"],
                "classes": dataset_params["classes"],
            },
            dataloader_params={"batch_size": BATCH_SIZE, "num_workers": 2},
        )

    train_params = {
        "silent_mode": True,
        "average_best_models": True,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 6e-5,
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
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
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

    trainer = Trainer(experiment_name=EXPERIMENT, ckpt_root_dir=CHECKPOINT_DIR)

    if train:

        trainer.train(
            model=model,
            training_params=train_params,
            train_loader=train_data,
            valid_loader=test_data,
        )

    if test:

        best_model = models.get(
            MODEL_ARCH,
            num_classes=NUM_CLASES,
            checkpoint_path=os.path.join(CHECKPOINT_DIR, EXPERIMENT, "ckpt_best.pth"),
        )

        trainer.test(
            model=best_model,
            test_loader=val_data,
            test_metrics_list=DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=len(dataset_params["classes"]),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7,
                ),
            ),
        )


if __name__ == "__main__":
    typer.run(main)

import glob
import os

import re

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Reshape,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Concatenate,
    Dense,
)
from tensorflow.keras.models import Model
from keras.utils import Sequence
import typer
import scipy
import keras

if tf.test.gpu_device_name():
    print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
else:
    print("No GPU available.")


class MatrixErrorMetrics:
    @staticmethod
    def mean_squared_error_matrix(y_true, y_pred):
        # Calculate the mean squared error for each element of the matrices
        mse_per_element = tf.reduce_mean(tf.square(y_true - y_pred), axis=(1, 2))
        # Average the MSE across all elements
        return tf.reduce_mean(mse_per_element)

    @staticmethod
    def mean_absolute_error_matrix(y_true, y_pred):
        # Calculate the mean absolute error for each element of the matrices
        mae_per_element = tf.reduce_mean(tf.abs(y_true - y_pred), axis=(1, 2))
        # Average the MAE across all elements
        return tf.reduce_mean(mae_per_element)

    @staticmethod
    def root_mean_squared_error_matrix(y_true, y_pred):
        # Calculate the root mean squared error for each element of the matrices
        rmse_per_element = tf.sqrt(
            tf.reduce_mean(tf.square(y_true - y_pred), axis=(1, 2))
        )
        # Average the RMSE across all elements
        return tf.reduce_mean(rmse_per_element)


class HomographyModel:
    def __init__(self, input_shape=(540, 540, 3)):
        self.input_shape = input_shape
        self.metrics = [
            MatrixErrorMetrics.mean_squared_error_matrix,
            MatrixErrorMetrics.mean_absolute_error_matrix,
            MatrixErrorMetrics.root_mean_squared_error_matrix,
        ]
        self.model = self._build_model()

    def _build_model(self):
        pretrained_model = tf.keras.applications.EfficientNetV2S(
            weights="imagenet", include_top=False, input_shape=self.input_shape
        )
        for layer in pretrained_model.layers:
            layer.trainable = True

        image_flatten_layer = Flatten()(pretrained_model.output)

        input_layer = image_flatten_layer
        inputs_ = pretrained_model.input

        dense_layer_1 = Dense(36, activation="relu")(input_layer)
        dense_layer_2 = Dense(18, activation="relu")(dense_layer_1)
        dense_layer = Dense(9, activation="relu")(dense_layer_2)
        output_layer = Reshape((3, 3))(dense_layer)

        model = Model(
            inputs=inputs_,
            outputs=output_layer,
        )
        return model

    def train(self, epochs: int, batch_size: int, train_dataset, validation_dataset):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00001),
            metrics=self.metrics,
            loss="mse",
        )

        checkpoint_callback = ModelCheckpoint(
            "./homography_checkpoints/best_ckpt.hdf5",
            monitor="val_loss",  # You can choose the metric to monitor
            save_best_only=True,  # Save only the best models
            save_weights_only=True,  # Save only the model weights
            mode="min",  # 'min' for loss minimization, 'max' for accuracy maximization
            verbose=1,  # Display messages when checkpoints are saved
        )

        early_stopping = EarlyStopping(monitor="val_loss", patience=10)

        reduce_lr = (
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=5, verbose=1, cooldown=10, min_lr=0.00001
            ),
        )

        self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            callbacks=[checkpoint_callback, early_stopping, reduce_lr],
        )

    def __call__(self, input_img):
        image = preprocess_input(
            cv2.resize(
                cv2.cvtColor(cv2.imread(input_img), cv2.COLOR_BGR2RGB),
                (self.input_shape[0], self.input_shape[1]),
            )
        )
        return self.model.predict(np.array([image]))

    def load_model(self, file):
        self.model.load_weights(file)


class HomographyDataset(Sequence):
    def __init__(self, input_paths, labels_list, image_size, batch_size):
        self.input_paths = input_paths
        self.labels_list = labels_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.indices = np.arange(len(labels_list))

    def __len__(self):
        return int(np.ceil(len(self.input_paths) / self.batch_size))

    def _read_homography(self, file_path: str):
        return np.load(file_path).reshape((3, 3))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        indices = self.indices[start:end]

        X = []
        y = []

        for i in indices:
            image = preprocess_input(
                cv2.resize(
                    cv2.cvtColor(cv2.imread(self.input_paths[i]), cv2.COLOR_BGR2RGB),
                    self.image_size,
                )
            )
            X.append(image)

            matrix = self._read_homography(self.labels_list[i])
            y.append(matrix)
            return np.array(X), np.array(y)


def get_number_from_string(s):
    s = s.split("/")[-1]
    match = re.search(r"\d+", s)
    if match:
        return int(match.group())
    else:
        return 0  # If no number is found, return 0


def main(train: bool = True, double_input: bool = False):
    DATA_DIR = (
        "/home/fer/Escritorio/futstatistics/datasets/narya/homography_dataset/dataset"
    )
    BATCH_SIZE = 8

    train_x = glob.glob(os.path.join(DATA_DIR, "train_img") + "/*.jpg")
    train_y = glob.glob(os.path.join(DATA_DIR, "train_homo") + "/*.npy")

    val_x = glob.glob(os.path.join(DATA_DIR, "test_img") + "/*.jpg")
    val_y = glob.glob(os.path.join(DATA_DIR, "test_homo") + "/*.npy")

    print(f"{len(train_x)} training images found")
    assert len(train_x) == len(train_y)

    print(f"{len(val_x)} validation images found")
    assert len(val_x) == len(val_y)

    # images = sorted(images_1 + images_2, key=get_number_from_string)
    # matrices = sorted(matrix_1 + matrix_2, key=get_number_from_string)

    # train_input, val_input, train_matrices, val_matrices = train_test_split(
    #     images, matrices, test_size=0.35
    # )

    train_dataset = HomographyDataset(
        train_x,
        train_y,
        image_size=(320, 320),
        batch_size=BATCH_SIZE,
    )
    test_dataset = HomographyDataset(
        val_x,
        val_y,
        image_size=(320, 320),
        batch_size=BATCH_SIZE,
    )

    # access first element of the dataset
    # input_, target_ = next(iter(train_dataset))
    # print(target_)

    model = HomographyModel(input_shape=(320, 320, 3))

    if train:
        model.train(
            epochs=100,
            batch_size=BATCH_SIZE,
            train_dataset=train_dataset,
            validation_dataset=test_dataset,
        )


if __name__ == "__main__":
    typer.run(main)

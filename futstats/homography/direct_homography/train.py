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
    def __init__(self, double_input, input_shape=(540, 540, 3)):
        self.input_shape = input_shape
        self.metrics = [
            MatrixErrorMetrics.mean_squared_error_matrix,
            MatrixErrorMetrics.mean_absolute_error_matrix,
            MatrixErrorMetrics.root_mean_squared_error_matrix,
        ]
        self.double_input = double_input
        self.model = self._build_model()

    def _build_model(self):
        pretrained_model = tf.keras.applications.EfficientNetV2S(
            weights="imagenet", include_top=False, input_shape=self.input_shape
        )
        for layer in pretrained_model.layers:
            layer.trainable = True

        image_flatten_layer = Flatten()(pretrained_model.output)

        if self.double_input:
            mask_input = Input(shape=(self.input_shape[0], self.input_shape[1], 1))
            mask_cnn = Conv2D(64, (3, 3), activation="relu")(mask_input)
            mask_cnn = MaxPooling2D((2, 2))(mask_cnn)
            mask_output = Flatten()(mask_cnn)
            input_layer = Concatenate()([image_flatten_layer, mask_output])
            inputs_ = [pretrained_model.input, mask_input]
        else:
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
            optimizer="adam",
            loss="mean_squared_error",
            metrics=self.metrics,
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

        self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            callbacks=[checkpoint_callback, early_stopping],
        )

    def __call__(self, input_img):
        image = preprocess_input(
            cv2.resize(
                cv2.cvtColor(cv2.imread(input_img), cv2.COLOR_BGR2RGB),
                self.image_size,
            )
        )
        return self.model.predict(image)


class HomographyDataset(Sequence):
    def __init__(self, input_paths, labels_list, double_input, image_size, batch_size):
        self.input_paths = input_paths
        self.labels_list = labels_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.double_input = double_input
        self.indices = np.arange(len(labels_list))

    def __len__(self):
        return int(np.ceil(len(self.input_paths) / self.batch_size))

    def _read_homography(self, file_path: str):
        homography_matrix = []

        with open(file_path, "r") as file:
            for line in file:
                elements = line.strip().split()

                row = [float(element) for element in elements]
                homography_matrix.append(row)

        return np.array(homography_matrix).reshape(3, 3)

    def _read_mat_mask(self, file_path: str):
        mat_contents = scipy.io.loadmat(file_path)
        return mat_contents["grass"]

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        indices = self.indices[start:end]

        if self.double_input:
            X_images = []
            X_masks = []
        else:
            X = []
        y = []

        for i in indices:
            if self.double_input:
                image = preprocess_input(
                    cv2.resize(
                        cv2.cvtColor(
                            cv2.imread(self.input_paths[i][0]), cv2.COLOR_BGR2RGB
                        ),
                        self.image_size,
                    )
                )
                X_images.append(image)
                mask = self._read_mat_mask(self.input_paths[i][1])
                X_masks.append(mask)
            else:
                image = preprocess_input(
                    cv2.resize(
                        cv2.cvtColor(
                            cv2.imread(self.input_paths[i]), cv2.COLOR_BGR2RGB
                        ),
                        self.image_size,
                    )
                )
                X.append(image)

            matrix = self._read_homography(self.labels_list[i])
            y.append(matrix)

        if self.double_input:
            return [np.array(X_images), np.array(X_masks)], np.array(y)
        else:
            return np.array(X), np.array(y)


def get_number_from_string(s):
    s = s.split("/")[-1]
    match = re.search(r"\d+", s)
    if match:
        return int(match.group())
    else:
        return 0  # If no number is found, return 0


def main(train: bool = True, double_input: bool = False):
    DATA_DIR = "/home/fer/Escritorio/futstatistics/datasets/dataset7_homography"
    BATCH_SIZE = 8

    images_1 = glob.glob(os.path.join(DATA_DIR, "test") + "/*.jpg")
    images_2 = glob.glob(os.path.join(DATA_DIR, "train_val") + "/*.jpg")

    matrix_1 = glob.glob(os.path.join(DATA_DIR, "test") + "/*.homographyMatrix")
    matrix_2 = glob.glob(os.path.join(DATA_DIR, "train_val") + "/*.homographyMatrix")

    images = sorted(images_1 + images_2, key=get_number_from_string)
    matrices = sorted(matrix_1 + matrix_2, key=get_number_from_string)

    if double_input:
        mask_1 = glob.glob(os.path.join(DATA_DIR, "test") + "/*.mat")
        mask_2 = glob.glob(os.path.join(DATA_DIR, "train_val") + "/*.mat")
        masks = sorted(mask_1 + mask_2, key=get_number_from_string)
        inputs = [(image, mask) for image, mask in zip(images, masks)]
    else:
        inputs = images

    train_input, val_input, train_matrices, val_matrices = train_test_split(
        inputs, matrices, test_size=0.35
    )

    train_dataset = HomographyDataset(
        train_input,
        train_matrices,
        double_input=double_input,
        image_size=(540, 540),
        batch_size=BATCH_SIZE,
    )
    test_dataset = HomographyDataset(
        val_input,
        val_matrices,
        double_input=double_input,
        image_size=(540, 540),
        batch_size=BATCH_SIZE,
    )

    model = HomographyModel(double_input=double_input, input_shape=(540, 540, 3))

    if train:
        model.train(
            epochs=100,
            batch_size=BATCH_SIZE,
            train_dataset=train_dataset,
            validation_dataset=test_dataset,
        )


if __name__ == "__main__":
    typer.run(main)

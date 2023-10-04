import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Reshape
import glob
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np

import tensorflow as tf

if tf.test.gpu_device_name():
    print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
else:
    print("No GPU available.")


pretrained_model = tf.keras.applications.EfficientNetV2S(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
for layer in pretrained_model.layers:
    layer.trainable = False


# Flatten the output of the pretrained model
flatten_layer = Flatten()(pretrained_model.output)

# Add a dense layer with 9 units (3x3 matrix)
dense_layer = Dense(9, activation="relu")(flatten_layer)

# Reshape the output to be a 3x3 matrix
output_layer = Reshape((3, 3))(dense_layer)

input_layer = pretrained_model.input
model = Model(
    inputs=input_layer,
    outputs=output_layer,
)

model.compile(optimizer="adam", loss="mean_squared_error")

DATA_DIR = "/home/fer/Escritorio/futstatistics/datasets/dataset7_homography"

DATA_DIR = "/home/fer/Escritorio/futstatistics/datasets/dataset7_homography"


images_1 = glob.glob(os.path.join(DATA_DIR, "test") + "/*.jpg")
images_2 = glob.glob(os.path.join(DATA_DIR, "train_val") + "/*.jpg")

labels_1 = glob.glob(os.path.join(DATA_DIR, "test") + "/*.homographyMatrix")
labels_2 = glob.glob(os.path.join(DATA_DIR, "train_val") + "/*.homographyMatrix")

images = sorted(images_1 + images_2)
labels = sorted(labels_1 + labels_2)

print(len(images))
print(len(labels))


# Define your batch size and number of epochs
batch_size = 4
epochs = 100


def read_homography(file_path):
    # Initialize an empty list to store the matrix data
    homography_matrix = []

    # Open the file and read its contents line by line
    with open(file_path, "r") as file:
        for line in file:
            # Split each line into individual elements (assuming space-separated values)
            elements = line.strip().split()

            # Convert the elements to float and append them to the matrix
            row = [float(element) for element in elements]
            homography_matrix.append(row)

    # Convert the list of lists to a NumPy array
    return np.array(homography_matrix)


def data_generator(images, matrices, batch_size):
    num_samples = len(images)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            batch_images = [
                preprocess_input(
                    cv2.resize(
                        cv2.cvtColor(cv2.imread(images[i]), cv2.COLOR_BGR2RGB),
                        (224, 224),
                    )
                )
                for i in batch_indices
            ]
            batch_matrices = [
                read_homography(matrices[i]).reshape(3, 3) for i in batch_indices
            ]

            # Preprocess your images and matrices here as needed

            yield np.array(batch_images), np.array(batch_matrices)


train_images, val_images, train_matrices, val_matrices = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

checkpoint_callback = ModelCheckpoint(
    "./homography_checkpoints/best_ckpt.hdf5",
    monitor="val_loss",  # You can choose the metric to monitor
    save_best_only=True,  # Save only the best models
    save_weights_only=True,  # Save only the model weights
    mode="min",  # 'min' for loss minimization, 'max' for accuracy maximization
    verbose=1,  # Display messages when checkpoints are saved
)

# Create the data generator
train_generator = data_generator(train_images, train_matrices, batch_size)
validation_generator = data_generator(val_images, val_matrices, batch_size)

# Train the model
# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(train_images) // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(val_images) // batch_size,
    callbacks=[checkpoint_callback],
)

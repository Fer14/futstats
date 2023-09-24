import numpy as np
import tensorflow as tf
import segmentation_models as sm
import cv2


def _build_keypoint_preprocessing(input_shape, backbone):
    """Builds the preprocessing function for the Field Keypoint Detector Model."""
    sm_preprocessing = sm.get_preprocessing(backbone)

    def preprocessing(input_img, **kwargs):

        to_normalize = False if np.percentile(input_img, 98) > 1.0 else True

        if len(input_img.shape) == 4:
            print(
                "Only preprocessing single image, we will consider the first one of the batch"
            )
            image = input_img[0] * 255.0 if to_normalize else input_img[0] * 1.0
        else:
            image = input_img * 255.0 if to_normalize else input_img * 1.0

        image = cv2.resize(image, input_shape)
        image = sm_preprocessing(image)
        return image

    return preprocessing


class KeypointDetectorModel:
    """Class for Keras Models to predict the keypoint in an image. These keypoints can then be used to
    compute the homography.

    Arguments:
        backbone: String, the backbone we want to use
        model_choice: The model architecture. ('FPN','Unet','Linknet')
        num_classes: Integer, number of mask to compute (= number of keypoints)
        input_shape: Tuple, shape of the model's input
    Call arguments:
        input_img: a np.array of shape input_shape
    """

    def __init__(
        self,
        backbone="efficientnetb3",
        model_choice="FPN",
        num_classes=29,
        input_shape=(320, 320),
    ):

        self.input_shape = input_shape
        self.classes = [str(i) for i in range(num_classes)] + ["background"]
        self.backbone = backbone

        n_classes = len(self.classes)
        activation = "softmax"

        if model_choice == "FPN":
            self.model = sm.FPN(
                self.backbone,
                classes=n_classes,
                activation=activation,
                input_shape=(input_shape[0], input_shape[1], 3),
                encoder_weights="imagenet",
            )
        else:
            self.model = None
            print("{} is not used yet".format(model_choice))

        self.preprocessing = _build_keypoint_preprocessing(input_shape, backbone)

    def __call__(self, input_img):

        img = self.preprocessing(input_img)
        pr_mask = self.model.predict(np.array([img]))
        return pr_mask

    def load_weights(self, weights_path):
        try:
            self.model.load_weights(weights_path)
            print("Succesfully loaded weights from {}".format(weights_path))
        except:
            orig_weights = "from Imagenet"
            print(
                "Could not load weights from {}, weights will be loaded {}".format(
                    weights_path, orig_weights
                )
            )

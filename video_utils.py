from typing import Generator, List
from dataclasses import dataclass
from anns import Detection
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


def generate_frames(video_file: str) -> Generator[np.ndarray, None, None]:
    video = cv2.VideoCapture(video_file)

    while video.isOpened():
        success, frame = video.read()

        if not success:
            break

        yield frame

    video.release()


def plot_image(
    image: np.ndarray, save: bool = False, filename: str = "crops_3/image.png"
) -> None:
    fig, ax = plt.subplots()
    ax.imshow(image[..., ::-1])
    plt.show()
    if save:
        ax.axis("off")
        ax.set_aspect("auto")
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)


def save_detections(image: np.ndarray, detections: List[Detection]):
    for i, detection in enumerate(detections):
        x2, y2 = detection.rect.bottom_right.int_xy_tuple
        x1, y1 = detection.rect.top_left.int_xy_tuple
        crop = image[y1:y2, x1:x2, :]
        plot_image(crop, True, f"./crops_2/{i}.png")


# stores information about output video file, width and height of the frame must be equal to input video
@dataclass(frozen=True)
class VideoConfig:
    fps: float
    width: int
    height: int


# create cv2.VideoWriter object that we can use to save output video
def get_video_writer(
    target_video_path: str, video_config: VideoConfig
) -> cv2.VideoWriter:
    video_target_dir = os.path.dirname(os.path.abspath(target_video_path))
    os.makedirs(video_target_dir, exist_ok=True)
    return cv2.VideoWriter(
        target_video_path,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=video_config.fps,
        frameSize=(video_config.width, video_config.height),
        isColor=True,
    )

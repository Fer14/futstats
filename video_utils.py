from typing import Generator
from dataclasses import dataclass

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
    image: np.ndarray, size: int = 12, save: bool = False, filename: str = "image.png"
) -> None:
    fig, ax = plt.subplots()
    ax.imshow(image[..., ::-1])
    plt.show()
    if save:
        ax.axis("off")
        ax.set_aspect("auto")
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)


"""
usage example:

video_config = VideoConfig(
    fps=30, 
    width=1920, 
    height=1080)
video_writer = get_video_writer(
    target_video_path=TARGET_VIDEO_PATH, 
    video_config=video_config)

for frame in frames:
    ...
    video_writer.write(frame)
    
video_writer.release()
"""


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

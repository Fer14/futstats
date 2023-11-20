import typer
import torch

from scripts.hsv_team_posession_tracking import launch_hsv_team_posession_tracking
from scripts.ball_path import launch_ball_homography
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


app = typer.Typer()


@app.command()
def hsv_team_posession_tracking():
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", "../models/yoloV5model", device=0
    )
    os.makedirs("./output_video/hsv_team_posession_tracking/", exist_ok=True)
    launch_hsv_team_posession_tracking(
        yoloNas=False,
        model=model,
        source_video_path="../../clips/0bfacc_0.mp4",
        target_video_path="./output_video/hsv_team_posession_tracking/0bfacc_0.mp4",
    )


@app.command()
def ball_homography():
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", "../models/yoloV5model", device=0
    )
    os.makedirs("./output_video/hsv_team_posession_tracking/", exist_ok=True)
    launch_ball_homography(
        yoloNas=False,
        model=model,
        field_model_path="../models/ckpt_best_nov_new.pth",
        field_img_path="./homography/images/field_2d.jpg",
        source_video_path="../../clips/0a2d9b_0.mp4",
        target_video_path="./output_video/homography_ball/0a2d9b_0.mp4",
        target_warped_video_path="./output_video/homography_ball/0a2d9b_0_warped.mp4",
        target_ball_track_path="./output_video/homography_ball/0a2d9b_0_ball.png",
    )


if __name__ == "__main__":
    app()

import typer
import torch

from scripts.hsv_team_posession_tracking import launch_hsv_team_posession_tracking
from scripts.ball_path import launch_ball_homography
from scripts.field_tracking import launch_field_tracking
from scripts.ball_tracking import launch_ball_tracking

# from scripts.field_tracking_roboflow import launch_field_tracking_roboflow

# from scripts.ball_path_roboflow import launch_ball_path_roboflow
from scripts.ball_path_narya import launch_ball_path_narya
import logging
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"
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
    os.makedirs("./output_video/homography_ball/", exist_ok=True)
    launch_ball_homography(
        yoloNas=False,
        model=model,
        field_model_path="../models/ckpt_best_dec.pth",
        field_img_path="./homography/images/field_2d.jpg",
        source_video_path="../../clips/08fd33_0.mp4",
        target_video_path="./output_video/homography_ball/08fd33_0.mp4",
        target_warped_video_path="./output_video/homography_ball/08fd33_0._warped.mp4",
        target_ball_track_path="./output_video/homography_ball/08fd33_0._ball.png",
    )


@app.command()
def field_tracking():
    os.makedirs("./output_video/field_tracking/", exist_ok=True)

    launch_field_tracking(
        field_model_path="../models/ckpt_best_dec.pth",
        field_img_path="./homography/images/field_2d.jpg",
        source_video_path="../../clips/0a2d9b_0.mp4",
        target_video_path="./output_video/field_tracking/0a2d9b_0.mp4",
    )


@app.command()
def ball_path_narya():
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", "../models/yoloV5model", device=0
    )
    os.makedirs("./output_video/ball_path_narya/", exist_ok=True)
    launch_ball_path_narya(
        yoloNas=False,
        model=model,
        field_model="./homography/narya_test/keypoints/keypoint_detector.h5",
        field_img_path="./homography/narya_test/template.png",
        source_video_path="../../clips/0a2d9b_0.mp4",
        target_video_path="./output_video/ball_path_narya/0a2d9b_0.mp4",
        target_warped_video_path="./output_video/ball_path_narya/0a2d9b_0_warped.mp4",
        target_ball_track_path="./output_video/ball_path_narya/0a2d9b_0_ball.png",
    )


# @app.command()
# def field_tracking_roboflow():
#     os.makedirs("./output_video/field_tracking_roboflow/", exist_ok=True)
#     launch_field_tracking_roboflow(
#         field_img_path="./homography/images/field_2d.jpg",
#         source_video_path="../../clips/0a2d9b_0.mp4",
#         target_video_path="./output_video/field_tracking_roboflow/0a2d9b_0.mp4",
#     )


# @app.command()
# def ball_path_roboflow():
#     model = torch.hub.load(
#         "ultralytics/yolov5", "custom", "../models/yoloV5model", device=0
#     )
#     os.makedirs("./output_video/ball_path_roboflow/", exist_ok=True)
#     launch_ball_path_roboflow(
#         yoloNas=False,
#         model=model,
#         field_img_path="./homography/images/field_2d.jpg",
#         source_video_path="../../clips/0a2d9b_0.mp4",
#         target_video_path="./output_video/ball_path_roboflow/0a2d9b_0.mp4",
#         target_warped_video_path="./output_video/ball_path_roboflow/0a2d9b_0_warped.mp4",
#         target_ball_track_path="./output_video/ball_path_roboflow/0a2d9b_0_ball.png",
#     )


@app.command()
def ball_tracking():
    os.makedirs("./output_video/ball_tracking/", exist_ok=True)
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", "../models/yoloV5model", device=0
    )
    launch_ball_tracking(
        yoloNas=False,
        model=model,
        source_video_path="../../clips/0a2d9b_0.mp4",
        target_video_path="./output_video/field_tracking/0a2d9b_0.mp4",
    )


if __name__ == "__main__":
    app()

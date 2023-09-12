import typer
import torch
from scripts.ball_posession import launch_ball_posession
from scripts.tracking import launch_tracking
from scripts.player_detection import launch_player_detection
from scripts.together import launch_all_together
from scripts.team_detection import launch_team_detection
from scripts.team_posession_detection import launch_team_posession_detection
from scripts.team_posession_tracking import launch_team_posession_tracking
from scripts.ensemble_team_posession_tracking import (
    launch_ensemble_team_posession_tracking,
)
import logging

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


app = typer.Typer()


@app.command()
def ball_possesion():
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", "./models/yoloV5model", device=0
    )
    launch_ball_posession(
        yoloNas=False,
        model=model,
        source_video_path="../clips/08fd33_4.mp4",
        target_video_path="./output_video/ball-possession/8fd33_4.mp4",
    )


@app.command()
def tracking():
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", "./models/yoloV5model", device=0
    )
    launch_tracking(
        yoloNas=False,
        model=model,
        source_video_path="../clips/08fd33_4.mp4",
        target_video_path="./output_video/tracking/8fd33_4.mp4",
    )


@app.command()
def player_detection():
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", "./models/yoloV5model", device=0
    )
    launch_player_detection(
        yoloNas=False,
        model=model,
        source_video_path="../clips/08fd33_4.mp4",
        target_video_path="./output_video/player_detection/8fd33_4.mp4",
    )


@app.command()
def all_together():
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", "./models/yoloV5model", device=0
    )
    launch_all_together(
        yoloNas=False,
        model=model,
        source_video_path="../clips/08fd33_4.mp4",
        target_video_path="./output_video/all_together/08fd33_4.mp4",
    )


@app.command()
def team_detection():
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", "./models/yoloV5model", device=0
    )
    launch_team_detection(
        yoloNas=False,
        model=model,
        source_video_path="../clips/08fd33_4.mp4",
        target_video_path="./output_video/team_detection/08fd33_4.mp4",
    )


@app.command()
def team_posession_detection():
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", "./models/yoloV5model", device=0
    )
    launch_team_posession_detection(
        yoloNas=False,
        model=model,
        source_video_path="../clips/08fd33_4.mp4",
        target_video_path="./output_video/team_posession_detection/08fd33_4.mp4",
    )


@app.command()
def team_posession_tracking():
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", "./models/yoloV5model", device=0
    )
    launch_team_posession_tracking(
        yoloNas=False,
        model=model,
        source_video_path="../clips/08fd33_4.mp4",
        target_video_path="./output_video/team_posession_tracking/8fd33_4.mp4",
    )


@app.command()
def ensemble_team_posession_tracking():
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", "./models/yoloV5model", device=0
    )
    launch_ensemble_team_posession_tracking(
        yoloNas=False,
        model=model,
        source_video_path="../clips/0bfacc_0.mp4",
        target_video_path="./output_video/ensemble_team_posession_tracking/0bfacc_0.mp4",
    )


if __name__ == "__main__":
    app()

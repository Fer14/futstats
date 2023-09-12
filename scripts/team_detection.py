from tqdm import tqdm
from video_utils import *
from anns import *


def launch_team_detection(
    yoloNas: bool,
    model,
    target_video_path: str,
    source_video_path: str,
):

    # initiate video writer
    video_config = VideoConfig(fps=30, width=1920, height=1080)
    video_writer = get_video_writer(
        target_video_path=target_video_path, video_config=video_config
    )

    # get fresh video frame generator
    frame_iterator = iter(generate_frames(video_file=source_video_path))

    # initiate annotators
    player_marker_annotator = MarkerTeamAnntator()

    # loop over frames
    for frame in tqdm(frame_iterator, total=750, desc="Annotating video"):

        # run detector
        if yoloNas:
            results = list(model.predict(frame, conf=0.25))[0]
            detections = Detection.from_yoloNas(pred=results)
        else:
            results = model(frame, size=1280)
            detections = Detection.from_yolo5(
                pred=results.pred[0].cpu().numpy(), names=model.names
            )

        goalkeeper_detections = filter_detections_by_class(
            detections=detections, class_name="goalkeeper"
        )
        player_detections = (
            filter_detections_by_class(detections=detections, class_name="player")
            + goalkeeper_detections
        )

        # annotate video frame
        annotated_image = frame.copy()

        annotated_image = player_marker_annotator.annotate(
            color=PLAYER_MARKER_FILL_COLOR,
            image=annotated_image,
            detections=player_detections,
        )

        # save video frame
        video_writer.write(annotated_image)

    # close output video
    video_writer.release()

from tqdm import tqdm
import sys

sys.path.append("..")
from video_utils import *
from anns import *
from posession_utils import *
from super_gradients.training import models
from mask_utils import get_perspective_transform
import torch
import logging

logging.disable(logging.INFO)

CLASSES = [
    "1",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "c",
]
NUM_CLASES = len(CLASSES)

POINT2POINT2D = {
    "1": (60, 60),
    "2": (310, 60),
    "3": (520, 60),
    "4": (850, 60),
    "5": (1050, 60),
    "6": (1300, 60),
    "7": (520, 160),
    "8": (850, 160),
    "9": (310, 360),
    "10": (550, 360),
    "11": (820, 360),
    "12": (1050, 360),
    "13": (60, 1020),
    "14": (680, 1020),
    "15": (860, 1020),
    "16": (1300, 1020),
    "c": (680, 850),
}


def launch_ball_path(
    yoloNas: bool,
    model,
    target_video_path: str,
    source_video_path: str,
):

    field = cv2.cvtColor(cv2.imread("field_2d.jpg"), cv2.COLOR_BGR2RGB)

    field_model = models.get(
        "yolo_nas_l",
        num_classes=NUM_CLASES,
        checkpoint_path="/home/fer/Escritorio/futstatistics/yolo_nas/checkpoints/FIELD_HOMOGRAPHY/ckpt_best.pth",
    )

    # initiate video writer
    video_config = VideoConfig(fps=30, width=1920, height=1080)
    video_writer = get_video_writer(
        target_video_path=target_video_path, video_config=video_config
    )

    # get fresh video frame generator
    frame_iterator = iter(generate_frames(video_file=source_video_path))

    ball_marker_annotator = BallAnntator()

    # loop over frames
    for iteration, frame in enumerate(tqdm(frame_iterator, total=750)):

        # run detector
        if yoloNas:
            results = list(model.predict(frame, conf=0.25))[0]
            detections = Detection.from_yoloNas(pred=results)
        else:
            results = model(frame, size=1280)
            detections = Detection.from_yolo5(
                pred=results.pred[0].cpu().numpy(), names=model.names
            )

        field_results = list(field_model.predict(frame, conf=0.25))[0]
        field_detections = Detection.from_yoloNas(pred=field_results)

        # filter detections by class
        ball_detections = filter_detections_by_class(
            detections=detections, class_name="ball"
        )

        src_points = []
        dst_points = []

        for detection in field_detections:
            x2, y2 = detection.rect.bottom_right.int_xy_tuple
            x1, y1 = detection.rect.top_left.int_xy_tuple
            # get the center of the box
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            if str(detection.class_name) in POINT2POINT2D:
                src_points.append(center)
                dst_points.append(POINT2POINT2D[str(detection.class_name)])

        pred_homo = get_perspective_transform(
            np.array(src_points), np.array(dst_points)
        )

        for ball_detection in ball_detections:
            x2, y2 = ball_detection.rect.bottom_right.int_xy_tuple
            x1, y1 = ball_detection.rect.top_left.int_xy_tuple
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            ball_pt = np.array([center], np.float32).reshape(-1, 1, 2)
            ball_pt_2d = cv2.perspectiveTransform(ball_pt, pred_homo)
            ball_pt_2d = ball_pt_2d.astype(int)
            field = cv2.circle(field, tuple(ball_pt_2d[0][0]), 10, (0, 0, 0), -1)

        # annotate video frame
        annotated_image = frame.copy()

        # Ball marker annotation
        annotated_image = ball_marker_annotator.annotate(
            image=annotated_image, detections=ball_detections
        )

        field_h = 410
        field_w = 273
        resized_field = cv2.resize(field, (field_w, field_h))
        h, w, _ = annotated_image.shape
        location_field_h = 1005
        location_field_w = 100
        annotated_image[
            h - location_field_h : h - (location_field_h - field_h),
            w - (field_w + location_field_w) : w - location_field_w,
            :,
        ] = resized_field
        # save video frame
        video_writer.write(annotated_image)

    # close output video
    video_writer.release()
    # save field image
    cv2.imwrite("../output_video/homography_ball/output.png", field)


# call launch_ball_path()
if __name__ == "__main__":
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", "../models/yoloV5model", device=0
    )
    os.makedirs("../output_video/homography_ball/", exist_ok=True)
    launch_ball_path(
        yoloNas=False,
        model=model,
        source_video_path="../../clips/0bfacc_0.mp4",
        target_video_path="../output_video/homography_ball/0bfacc_0.mp4",
    )

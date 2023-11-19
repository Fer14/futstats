from dataclasses import dataclass

import numpy as np
from annotations.anns import Detection
from norfair import Detection as nfDetection
from onemetric.cv.utils.iou import box_iou_batch
from yolox.tracker.byte_tracker import BYTETracker, STrack


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


"""
BYTETracker does not assign tracker_id to existing bounding boxes but rather
predicts the next bounding box position based on previous one. Therefore, we 
need to find a way to match our bounding boxes with predictions.

usage example:

byte_tracker = BYTETracker(BYTETrackerArgs())
for frame in frames:
    ...
    results = model(frame, size=1280)
    detections = Detection.from_results(
        pred=results.pred[0].cpu().numpy(), 
        names=model.names)
    ...
    tracks = byte_tracker.update(
        output_results=detections2boxes(detections=detections),
        img_info=frame.shape,
        img_size=frame.shape)
    detections = match_detections_with_tracks(detections=detections, tracks=tracks)
"""


# converts list[Detection] into format that can be consumed by match_detections_with_tracks function
def detections2boxes(
    detections: list[Detection], with_confidence: bool = True
) -> np.ndarray:
    return np.array(
        [
            [
                detection.rect.top_left.x,
                detection.rect.top_left.y,
                detection.rect.bottom_right.x,
                detection.rect.bottom_right.y,
                detection.confidence,
            ]
            if with_confidence
            else [
                detection.rect.top_left.x,
                detection.rect.top_left.y,
                detection.rect.bottom_right.x,
                detection.rect.bottom_right.y,
            ]
            for detection in detections
        ],
        dtype=float,
    )


# def detections2norfairDetection(
#     detections: list[Detection], with_confidence: bool = True
# ) -> np.ndarray:

#     norfairDetections = []

#     for detection in detections:

#         x2, y2 = detection.rect.bottom_right.int_xy_tuple
#         x1, y1 = detection.rect.top_left.int_xy_tuple
#         center_x = int((x1 + x2) / 2)
#         center_y = int((y1 + y2) / 2)

#         norfairDetection = nfDetection(points=np.array([center_x, center_y]))
#         norfairDetections.append(norfairDetection)

#     return norfairDetections


# converts list[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: list[STrack]) -> np.ndarray:
    return np.array([track.tlbr for track in tracks], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: list[Detection], tracks: list[STrack]
) -> list[Detection]:
    detection_boxes = detections2boxes(detections=detections, with_confidence=False)
    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detection_boxes)
    track2detection = np.argmax(iou, axis=1)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            detections[detection_index].tracker_id = tracks[tracker_index].track_id
    return detections

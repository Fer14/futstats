# Fμτsτατs

![logo](../imgs/logos/logo_black2.png)

Football Statistics generator with Python

# CODE STRUCTURE


## LAUNCH

The script is a command-line application utilizing the Typer library to perform two distinct tasks related to video analysis:

- HSV Team Possession Tracking:
    Loads a custom YOLOv5 model using PyTorch from the specified path.
    Executes the launch_hsv_team_posession_tracking function, performing team possession tracking using computer vision techniques like YOLO object detection on a specified input video.
    Saves the output video with the tracked possessions to the designated output path.

- Ball Homography:
    Similar to the first command, it loads a custom YOLOv5 model using PyTorch from the given path.
    Executes the launch_ball_homography function, aimed at performing ball tracking utilizing homography and computer vision methods on a specified input video.
    Saves various outputs including the resulting video with ball tracking, a warped version of the video showing the ball's perspective, and an image highlighting the ball's trajectory.

The script sets up a command-line interface using Typer, allowing users to execute these tasks separately by running the script with the desired command. These functionalities are centered around computer vision tasks, employing deep learning-based object detection (YOLOv5) and homography-based methods for analyzing video content.


## ANNOTATIONS

The provided code comprises utility functions and data structures for annotating object detections in images. It includes classes like Point, Rect, and Detection to represent geometric shapes and detected objects. A set of annotation utilities allows drawing shapes like circles, rectangles, ellipses, and polygons on images based on detection information. Various annotator classes, such as MarkerAnntator, LandmarkAnntator, and BaseAnnotator, handle annotation tasks by providing methods to visualize detections with different shapes and colors. Additionally, the code contains definitions for colors, markers, and methods to annotate possession statistics, aiding in visual analysis and interpretation of object detection results. The FieldAnnotator class facilitates the annotation and update of a field in images based on detections, using homography transformation for positioning. Overall, this code collection serves as a versatile toolkit for annotating and visualizing object detections in computer vision applications.

## HOMOGRAPHY

The homography utilities module contains functions to perform homography transformations on images and points.

It it divided in four main folders:

- Direct homography: Neccesary code to train a deep learning model that given an image, returns its Homography Matrix
- Keypoint homography: Neccesary code to train a deep learning model that given an image, returns 28 football field points to use to compute the Homography Matrix
- Keypoint homography: Neccesary code to train a deep learning model that given an image, returns 14 football field points (orientation agnostic) to use to compute the Homography Matrix
- Experiments: Folder used to do experiments. POr example cv2.Houghlines has been used for experimentation without any good result


## POSESSION

Contains classes and functions related to analyzing player positions, calculating possession, and clustering colors and detections in sports-related computer vision tasks, in the context of tracking players and ball possession in a football game.

## SCRIPTS

This module contains the necessary scripts ```HSV Team Possession Tracking``` and ```Ball Homography```


## TRACKING

The tracking utilities code is focused on integrating the BYTETracker, aiding in associating bounding boxes between consecutive frames in a video sequence:

## VIDEO

The video utilities module offers functionalities to handle video processing tasks using OpenCV and Matplotlib in Python:


![logo](../imgs/logos/logo_white.png)

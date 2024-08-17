# Fμτsτατs

![logo](imgs/logos/logo_black2.png)

Football Statistics generator with Python

## Introduction

This Python code provides a basic statistic/visualization generator than can be use to asses the percentage of posession for each team, the heatmap of the ball and diferent players. It expandes the roboflow player object dection. Its is not perfect and was just intended as a fun personal project.

## Ball posession

[08fd33_4.webm](https://github.com/user-attachments/assets/3d2b9577-7980-4067-8e6b-4d14344a85b4)


It uses a ```YOLOV8``` model to detect players in every frame, a tracking ```Bytrack``` model to asign a id to every player and a ```KNearestNeighbour``` algorithm to cluster the players in two teams based on the HSV of the object detection bounding box.

## Ball tracking across the field

[08fd33_0.webm](https://github.com/user-attachments/assets/ba3b610f-b603-4c36-bee2-4412fd590e22)



It uses the previous ```YOLOV8``` model to predict the ball and a new model to predict the keypoints of the football field needed to compute the homography matrix.

## Bird view


[08fd33_0_warped.webm](https://github.com/user-attachments/assets/cf04242e-840c-4f16-887e-0e0b4f2f47d1)


It uses the homography matrix to compute the bird view of the football field and the ball trajectory using some ```open-cv``` modules.


## Code structure

An overview of the code structure can be found [here](futstats/README.md)


## Instalation

- Clone ByteTrack
- Clone Yolov5
- Install project with :
```python
pip install -e .
```

## Usage

```python
python launch.py hsv-team-posession-tracking
```
or
```python
python launch.py ball-homography
```


![logo](imgs/logos/logo_white.png)

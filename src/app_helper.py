import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
from .config import shooting_result
import sys
from sys import platform
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from .utils import detect_shot, detect_image, detect_API, yolo_init, yolo_pose_init
from statistics import mean

def getVideoStream(video_path):
    global shooting_result
    shooting_result.update({
        'attempts': 0,
        'made': 0,
        'miss': 0,
        'avg_elbow_angle': 0,
        'avg_knee_angle': 0,
        'avg_release_angle': 0,
        'avg_ballInHand_time': 0,
        'frame_data': [],
        'shots': []
    })
    
    yolo_pose_model = yolo_pose_init()
    yolo_model = yolo_init()

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        fps = 30.0
    trace = np.full((int(height), int(width), 3), 255, np.uint8)

    fig = plt.figure()
    #objects to store detection status
    previous = {
    'ball': np.array([0, 0]),  # x, y
    'hoop': np.array([0, 0, 0, 0]),  # xmin, ymax, xmax, ymin
        'hoop_height': 0
    }
    during_shooting = {
        'isShooting': False,
        'balls_during_shooting': [],
        'release_angle_list': [],
        'release_point': []
    }
    shooting_pose = {
        'ball_in_hand': False,
        'elbow_angle': 370,
        'knee_angle': 370,
        'ballInHand_frames': 0,
        'elbow_angle_list': [],
        'knee_angle_list': [],
        'ballInHand_frames_list': []
    }
    shot_result = {
        'displayFrames': 0,
        'release_displayFrames': 0,
        'judgement': ""
    }

    skip_count = 0
    while True:
        ret, img = cap.read()
        if ret == False:
            break
        skip_count += 1
        if(skip_count < 4):
            continue
        skip_count = 0
        detection, trace = detect_shot(img, trace, width, height, yolo_model, previous, during_shooting, shot_result, fig, yolo_pose_model, shooting_pose)

        detection = cv2.resize(detection, (0, 0), fx=0.83, fy=0.83)
        if 'out_writer' not in shot_result:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            shot_result['out_writer'] = cv2.VideoWriter('./static/detections/raw_output.mp4', fourcc, fps / 4.0, (detection.shape[1], detection.shape[0]))
        shot_result['out_writer'].write(detection)
        frame = cv2.imencode('.jpg', detection)[1].tobytes()
        result = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        yield result


    # getting average shooting angle
    shooting_result['avg_elbow_angle'] = round(mean(shooting_pose['elbow_angle_list']) if shooting_pose['elbow_angle_list'] else 0, 2)
    shooting_result['avg_knee_angle'] = round(mean(shooting_pose['knee_angle_list']) if shooting_pose['knee_angle_list'] else 0, 2)
    shooting_result['avg_release_angle'] = round(mean(during_shooting['release_angle_list']) if during_shooting['release_angle_list'] else 0, 2)
    shooting_result['avg_ballInHand_time'] = round((mean(shooting_pose['ballInHand_frames_list']) if shooting_pose['ballInHand_frames_list'] else 0) * (4 / fps), 2)

    print("avg", shooting_result['avg_elbow_angle'])
    print("avg", shooting_result['avg_knee_angle'])
    print("avg", shooting_result['avg_release_angle'])
    print("avg", shooting_result['avg_ballInHand_time'])

    plt.title("Trajectory Fitting", figure=fig)
    plt.ylim(bottom=0, top=height)
    trajectory_path = os.path.join(
        os.getcwd(), "static/detections/trajectory_fitting.jpg")
    fig.savefig(trajectory_path)
    fig.clear()
    trace_path = os.path.join(os.getcwd(), "static/detections/basketball_trace.jpg")
    cv2.imwrite(trace_path, trace)
    if 'out_writer' in shot_result and shot_result['out_writer'] is not None:
        shot_result['out_writer'].release()
        os.system("ffmpeg -y -i ./static/detections/raw_output.mp4 -vcodec libx264 ./static/detections/final_output.mp4")

def get_image(image_path, img_name, response):
    output_path = './static/detections/'
    # reading the images & apply detection 
    image = cv2.imread(image_path)
    filename = img_name
    detection = detect_image(image, response)

    cv2.imwrite(output_path + '{}' .format(filename), detection)
    print('output saved to: {}'.format(output_path + '{}'.format(filename)))

def detectionAPI(response, image_path):
    image = cv2.imread(image_path)
    detect_API(response, image)

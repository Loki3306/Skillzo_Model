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
tf.disable_v2_behavior()

# ─── Skillzo Brand Palette (BGR) ───────────────────────────────────────────
SKZ_ORANGE   = (0,   107, 255)   # #FF6B00  — primary accent
SKZ_WHITE    = (255, 255, 255)
SKZ_BLACK    = (0,   0,   0)
SKZ_DARK     = (20,  20,  20)    # near-black for badge backgrounds
SKZ_GREEN    = (80,  200, 80)    # made shots
SKZ_RED      = (60,  60,  220)   # missed shots (BGR: red)
SKZ_CYAN     = (220, 220, 60)    # release angle highlight
SKZ_ALPHA    = 0.65              # overlay transparency
SKZ_FONT     = cv2.FONT_HERSHEY_DUPLEX

def skz_pill(frame, text, origin, font_scale=0.65, thickness=1,
              txt_color=SKZ_WHITE, bg_color=SKZ_DARK, accent=SKZ_ORANGE):
    """Draw a pill-shaped badge with a left accent bar."""
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, SKZ_FONT, font_scale, thickness)
    pad_x, pad_y = 10, 6
    bar_w = 4

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - th - pad_y),
                  (x + tw + pad_x * 2 + bar_w, y + baseline + pad_y),
                  bg_color, -1)
    cv2.addWeighted(overlay, SKZ_ALPHA, frame, 1 - SKZ_ALPHA, 0, frame)

    # Left accent bar
    cv2.rectangle(frame, (x, y - th - pad_y),
                  (x + bar_w, y + baseline + pad_y), accent, -1)

    # Text
    cv2.putText(frame, text, (x + bar_w + pad_x, y),
                SKZ_FONT, font_scale, txt_color, thickness, cv2.LINE_AA)


def skz_hud(frame, made, attempts):
    """Render a top-left scoreboard showing live Skillzo branding + score."""
    h, w = frame.shape[:2]
    bx, by, bw, bh = 12, 12, 210, 74

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), SKZ_DARK, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Top orange stripe
    cv2.rectangle(frame, (bx, by), (bx + bw, by + 4), SKZ_ORANGE, -1)

    # Brand name
    cv2.putText(frame, "SKILLZO AI", (bx + 10, by + 22),
                SKZ_FONT, 0.5, SKZ_ORANGE, 1, cv2.LINE_AA)

    # Score
    score_str = f"{made}/{attempts}"
    cv2.putText(frame, score_str, (bx + 10, by + 56),
                SKZ_FONT, 1.3, SKZ_WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, "MADE", (bx + 10 + len(score_str) * 18 + 8, by + 56),
                SKZ_FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def skz_glow_circle(frame, center, radius, color, thickness=-1):
    """Draw a circle with a subtle glow ring around it."""
    glow_color = tuple(min(int(c * 1.4), 255) for c in color)
    cv2.circle(frame, center, radius + 5, glow_color, 2, cv2.LINE_AA)
    cv2.circle(frame, center, radius, color, thickness, cv2.LINE_AA)
    if thickness == -1:
        cv2.circle(frame, center, max(radius - 4, 2), SKZ_WHITE, 2, cv2.LINE_AA)


def skz_judgement(frame, text, center_x, center_y):
    """Large animated-style SCORE / MISS judgement overlay."""
    color  = SKZ_GREEN if text == "SCORE" else SKZ_RED
    label  = "● " + text
    scale  = 2.2
    thick  = 5
    (tw, th), _ = cv2.getTextSize(label, SKZ_FONT, scale, thick)
    tx = max(center_x - tw // 2, 4)
    ty = max(center_y - 80, th + 4)

    # Shadow
    cv2.putText(frame, label, (tx + 3, ty + 3),
                SKZ_FONT, scale, SKZ_BLACK, thick + 2, cv2.LINE_AA)
    # Main text
    cv2.putText(frame, label, (tx, ty),
                SKZ_FONT, scale, color, thick, cv2.LINE_AA)

_cached_tf_session_vars = None
_cached_op_wrapper = None
_cached_op_datum = None

def tensorflow_init():
    global _cached_tf_session_vars
    if _cached_tf_session_vars is not None:
        return _cached_tf_session_vars
    MODEL_NAME = 'inference_graph'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    _cached_tf_session_vars = (detection_graph, image_tensor, boxes, scores, classes, num_detections)
    return _cached_tf_session_vars

def openpose_init():
    global _cached_op_wrapper, _cached_op_datum
    if _cached_op_wrapper is not None:
        return _cached_op_datum, _cached_op_wrapper
    try:
        if platform == "win32":
            sys.path.append(os.path.dirname(os.getcwd()))
            import OpenPose.Release.pyopenpose as op
        else:
            path = os.path.join(os.getcwd(), 'OpenPose/openpose')
            print(path)
            sys.path.append(path)
            import pyopenpose as op
    except ImportError as e:
        print("Something went wrong when importing OpenPose")
        raise e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "./OpenPose/models"
    params["num_gpu"] = 1
    params["num_gpu_start"] = 0

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    _cached_op_datum = op.Datum()
    _cached_op_wrapper = opWrapper
    return _cached_op_datum, _cached_op_wrapper

def fit_func(x, a, b, c):
    return a*(x ** 2) + b * x + c


def trajectory_fit(balls, height, width, shotJudgement, fig):
    x = [ball[0] for ball in balls]
    y = [height - ball[1] for ball in balls]

    try:
        params = curve_fit(fit_func, x, y)
        [a, b, c] = params[0]   
    except:
        print("fitting error")
        a = 0
        b = 0
        c = 0
    x_pos = np.arange(0, width, 1)
    y_pos = [(a * (x_val ** 2)) + (b * x_val) + c for x_val in x_pos]

    if(shotJudgement == "MISS"):
        plt.plot(x, y, 'ro', figure=fig)
        plt.plot(x_pos, y_pos, linestyle='-', color='red',
                 alpha=0.4, linewidth=5, figure=fig)
    else:
        plt.plot(x, y, 'go', figure=fig)
        plt.plot(x_pos, y_pos, linestyle='-', color='green',
                 alpha=0.4, linewidth=5, figure=fig)

def distance(x, y):
    return ((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2) ** (1/2)


def calculateAngle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return round(np.degrees(angle), 2)


def getAngleFromDatum(datum):
    hipX, hipY, _ = datum.poseKeypoints[0][9]
    kneeX, kneeY, _ = datum.poseKeypoints[0][10]
    ankleX, ankleY, _ = datum.poseKeypoints[0][11]

    shoulderX, shoulderY, _ = datum.poseKeypoints[0][2]
    elbowX, elbowY, _ = datum.poseKeypoints[0][3]
    wristX, wristY, _ = datum.poseKeypoints[0][4]

    kneeAngle = calculateAngle(np.array([hipX, hipY]), np.array(
        [kneeX, kneeY]), np.array([ankleX, ankleY]))
    elbowAngle = calculateAngle(np.array([shoulderX, shoulderY]), np.array(
        [elbowX, elbowY]), np.array([wristX, wristY]))

    elbowCoord = np.array([int(elbowX), int(elbowY)])
    kneeCoord = np.array([int(kneeX), int(kneeY)])
    return elbowAngle, kneeAngle, elbowCoord, kneeCoord

def detect_shot(frame, trace, width, height, sess, image_tensor, boxes, scores, classes, num_detections, previous, during_shooting, shot_result, fig, datum, opWrapper, shooting_pose):
    global shooting_result

    if(shot_result['displayFrames'] > 0):
        shot_result['displayFrames'] -= 1
    if(shot_result['release_displayFrames'] > 0):
        shot_result['release_displayFrames'] -= 1
    if(shooting_pose['ball_in_hand']):
        shooting_pose['ballInHand_frames'] += 1
        # print("ball in hand")

    # getting openpose keypoints
    datum.cvInputData = frame
    import pyopenpose as op
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    try:
        headX, headY, headConf = datum.poseKeypoints[0][0]
        handX, handY, handConf = datum.poseKeypoints[0][4]
        elbowAngle, kneeAngle, elbowCoord, kneeCoord = getAngleFromDatum(datum)
    except:
        print("Something went wrong with OpenPose")
        headX = 0
        headY = 0
        handX = 0
        handY = 0
        elbowAngle = 0
        kneeAngle = 0
        elbowCoord = np.array([0, 0])
        kneeCoord = np.array([0, 0])

    frame_expanded = np.expand_dims(frame, axis=0)
    # main tensorflow detection
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # ─── Skillzo overlay: stat badges + HUD ───────────────────────────────
    frame = datum.cvOutputData

    # Joint angle pill badges anchored to the joint positions
    skz_pill(frame, f'ELBOW  {elbowAngle:.1f}deg',
             (elbowCoord[0] + 18, elbowCoord[1]), accent=SKZ_ORANGE)
    skz_pill(frame, f'KNEE   {kneeAngle:.1f}deg',
             (kneeCoord[0] + 18, kneeCoord[1]), accent=SKZ_CYAN)

    # Release angle badge (shown for 30 frames after release)
    if shot_result['release_displayFrames']:
        rx = during_shooting['release_point'][0] - 80
        ry = during_shooting['release_point'][1] + 90
        skz_pill(frame, f'RELEASE  {during_shooting["release_angle_list"][-1]:.1f}deg',
                 (max(rx, 4), max(ry, 24)), font_scale=0.7, accent=SKZ_CYAN)

    # Live scoreboard HUD
    skz_hud(frame, shooting_result.get('made', 0), shooting_result.get('attempts', 0))

    for i, box in enumerate(boxes[0]):
        if (scores[0][i] > 0.2):
            ymin = int((box[0] * height))
            xmin = int((box[1] * width))
            ymax = int((box[2] * height))
            xmax = int((box[3] * width))
            xCoor = int(np.mean([xmin, xmax]))
            yCoor = int(np.mean([ymin, ymax]))
            # Basketball (not head)
            if(classes[0][i] == 1 and (distance([headX, headY], [xCoor, yCoor]) > 30)):
                if 'frame_data' not in shooting_result: shooting_result['frame_data'] = []
                shooting_result['frame_data'].append({"ball_x": float(xCoor), "ball_y": float(yCoor), "elbow_angle": float(elbowAngle), "knee_angle": float(kneeAngle)})

                # recording shooting pose
                if(distance([xCoor, yCoor], [handX, handY]) < 120):
                    shooting_pose['ball_in_hand'] = True
                    shooting_pose['knee_angle'] = min(
                        shooting_pose['knee_angle'], kneeAngle)
                    shooting_pose['elbow_angle'] = min(
                        shooting_pose['elbow_angle'], elbowAngle)
                else:
                    shooting_pose['ball_in_hand'] = False

                # During Shooting
                if(ymin < (previous['hoop_height'])):
                    if(not during_shooting['isShooting']):
                        during_shooting['isShooting'] = True

                    during_shooting['balls_during_shooting'].append(
                        [xCoor, yCoor])

                    #calculating release angle
                    if(len(during_shooting['balls_during_shooting']) == 2):
                        first_shooting_point = during_shooting['balls_during_shooting'][0]
                        release_angle = calculateAngle(np.array(during_shooting['balls_during_shooting'][1]), np.array(
                            first_shooting_point), np.array([first_shooting_point[0] + 1, first_shooting_point[1]]))
                        if(release_angle > 90):
                            release_angle = 180 - release_angle
                        during_shooting['release_angle_list'].append(
                            release_angle)
                        during_shooting['release_point'] = first_shooting_point
                        shot_result['release_displayFrames'] = 30
                        print("release angle:", release_angle)

                    # ── Ball-in-flight: orange glow dot ──────────────────
                    skz_glow_circle(frame, (xCoor, yCoor), 8, SKZ_ORANGE, thickness=-1)
                    skz_glow_circle(trace, (xCoor, yCoor), 8, SKZ_ORANGE, thickness=-1)

                # Not shooting
                elif(ymin >= (previous['hoop_height'] - 30) and (distance([xCoor, yCoor], previous['ball']) < 100)):
                    # the moment when ball go below basket
                    if(during_shooting['isShooting']):
                        if(xCoor >= previous['hoop'][0] and xCoor <= previous['hoop'][2]):  # shot
                            shooting_result['attempts'] += 1
                            shooting_result['made'] += 1
                            shot_result['displayFrames'] = 10
                            shot_result['judgement'] = "SCORE"
                            print("SCORE")
                            # ── SCORE trace: bright green thick line + dots ──
                            points = np.asarray(
                                during_shooting['balls_during_shooting'], dtype=np.int32)
                            cv2.polylines(trace, [points], False,
                                          color=SKZ_DARK, thickness=6, lineType=cv2.LINE_AA)
                            cv2.polylines(trace, [points], False,
                                          color=SKZ_GREEN, thickness=2, lineType=cv2.LINE_AA)
                            for ballCoor in during_shooting['balls_during_shooting']:
                                skz_glow_circle(trace, (ballCoor[0], ballCoor[1]), 7, SKZ_GREEN, -1)
                                
                            # ── Save clean shot image ──
                            os.makedirs('./static/detections/shots', exist_ok=True)
                            shot_img = np.full((int(height), int(width), 3), 255, np.uint8)
                            if 'hoop' in previous:
                                h_coords = [int(x) for x in previous['hoop']]
                                cv2.rectangle(shot_img, (h_coords[0], h_coords[1]), (h_coords[2], h_coords[3]), SKZ_DARK, 8)
                                cv2.rectangle(shot_img, (h_coords[0], h_coords[1]), (h_coords[2], h_coords[3]), SKZ_ORANGE, 3)
                            cv2.polylines(shot_img, [points], False, color=SKZ_DARK, thickness=6, lineType=cv2.LINE_AA)
                            cv2.polylines(shot_img, [points], False, color=SKZ_GREEN, thickness=2, lineType=cv2.LINE_AA)
                            for ballCoor in during_shooting['balls_during_shooting']:
                                skz_glow_circle(shot_img, (int(ballCoor[0]), int(ballCoor[1])), 7, SKZ_GREEN, -1)
                            cv2.imwrite(f'./static/detections/shots/shot_{shooting_result["attempts"]}.jpg', shot_img)
                        else:  # miss
                            shooting_result['attempts'] += 1
                            shooting_result['miss'] += 1
                            shot_result['displayFrames'] = 10
                            shot_result['judgement'] = "MISS"
                            print("miss")
                            # ── MISS trace: red thick line + dots ────────────
                            points = np.asarray(
                                during_shooting['balls_during_shooting'], dtype=np.int32)
                            cv2.polylines(trace, [points], False,
                                          color=SKZ_DARK, thickness=6, lineType=cv2.LINE_AA)
                            cv2.polylines(trace, [points], False,
                                          color=SKZ_RED, thickness=2, lineType=cv2.LINE_AA)
                            for ballCoor in during_shooting['balls_during_shooting']:
                                skz_glow_circle(trace, (ballCoor[0], ballCoor[1]), 7, SKZ_RED, -1)
                                
                            # ── Save clean shot image ──
                            os.makedirs('./static/detections/shots', exist_ok=True)
                            shot_img = np.full((int(height), int(width), 3), 255, np.uint8)
                            if 'hoop' in previous:
                                h_coords = [int(x) for x in previous['hoop']]
                                cv2.rectangle(shot_img, (h_coords[0], h_coords[1]), (h_coords[2], h_coords[3]), SKZ_DARK, 8)
                                cv2.rectangle(shot_img, (h_coords[0], h_coords[1]), (h_coords[2], h_coords[3]), SKZ_ORANGE, 3)
                            cv2.polylines(shot_img, [points], False, color=SKZ_DARK, thickness=6, lineType=cv2.LINE_AA)
                            cv2.polylines(shot_img, [points], False, color=SKZ_RED, thickness=2, lineType=cv2.LINE_AA)
                            for ballCoor in during_shooting['balls_during_shooting']:
                                skz_glow_circle(shot_img, (int(ballCoor[0]), int(ballCoor[1])), 7, SKZ_RED, -1)
                            cv2.imwrite(f'./static/detections/shots/shot_{shooting_result["attempts"]}.jpg', shot_img)

                        # reset all variables
                        trajectory_fit(
                            during_shooting['balls_during_shooting'], height, width, shot_result['judgement'], fig)
                        
                        if 'shots' not in shooting_result:
                            shooting_result['shots'] = []
                        
                        shooting_result['shots'].append({
                            'result': shot_result['judgement'],
                            'release_angle': float(during_shooting['release_angle_list'][-1]) if during_shooting['release_angle_list'] else 0.0,
                            'elbow_angle': float(shooting_pose['elbow_angle']),
                            'knee_angle': float(shooting_pose['knee_angle']),
                            'ballInHand_frames': int(shooting_pose['ballInHand_frames']),
                            'trajectory': [[int(x) for x in pt] for pt in during_shooting['balls_during_shooting']],
                            'hoop_bbox': [int(x) for x in previous['hoop']]
                        })

                        during_shooting['balls_during_shooting'].clear()
                        during_shooting['isShooting'] = False
                        shooting_pose['ballInHand_frames_list'].append(
                            shooting_pose['ballInHand_frames'])
                        print("ball in hand frames: ",
                              shooting_pose['ballInHand_frames'])
                        shooting_pose['ballInHand_frames'] = 0

                        print("elbow angle: ", shooting_pose['elbow_angle'])
                        print("knee angle: ", shooting_pose['knee_angle'])
                        shooting_pose['elbow_angle_list'].append(
                            shooting_pose['elbow_angle'])
                        shooting_pose['knee_angle_list'].append(
                            shooting_pose['knee_angle'])
                        shooting_pose['elbow_angle'] = 370
                        shooting_pose['knee_angle'] = 370

                    # ── Ball at rest: white-center orange dot ─────────────
                    skz_glow_circle(frame, (xCoor, yCoor), 10, SKZ_ORANGE, thickness=-1)
                    skz_glow_circle(trace, (xCoor, yCoor), 10, SKZ_ORANGE, thickness=-1)

                previous['ball'][0] = xCoor
                previous['ball'][1] = yCoor

            if(classes[0][i] == 2):  # Rim
                # ── Hoop box: clear old, draw new in Skillzo orange ───────
                cv2.rectangle(
                    trace, (previous['hoop'][0], previous['hoop'][1]),
                    (previous['hoop'][2], previous['hoop'][3]), SKZ_WHITE, 5)
                # Orange hoop rectangle with rounded feel (double-draw trick)
                cv2.rectangle(frame, (xmin, ymax), (xmax, ymin), SKZ_DARK, 8)
                cv2.rectangle(frame, (xmin, ymax), (xmax, ymin), SKZ_ORANGE, 3)
                cv2.rectangle(trace, (xmin, ymax), (xmax, ymin), SKZ_DARK, 8)
                cv2.rectangle(trace, (xmin, ymax), (xmax, ymin), SKZ_ORANGE, 3)

                # ── Judgement overlay ─────────────────────────────────────
                if shot_result['displayFrames']:
                    skz_judgement(frame, shot_result['judgement'], xCoor, yCoor)

                previous['hoop'][0] = xmin
                previous['hoop'][1] = ymax
                previous['hoop'][2] = xmax
                previous['hoop'][3] = ymin
                previous['hoop_height'] = max(ymin, previous['hoop_height'])

    combined = np.concatenate((frame, trace), axis=1)
    return combined, trace


def detect_image(img, response):
    height, width = img.shape[:2]
    detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.36

    with tf.Session(graph=detection_graph, config=config) as sess:
        img_expanded = np.expand_dims(img, axis=0)
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: img_expanded})
        valid_detections = 0

        for i, box in enumerate(boxes[0]):
            # print("detect")
            if (scores[0][i] > 0.2):
                valid_detections += 1
                ymin = int((box[0] * height))
                xmin = int((box[1] * width))
                ymax = int((box[2] * height))
                xmax = int((box[3] * width))
                xCoor = int(np.mean([xmin, xmax]))
                yCoor = int(np.mean([ymin, ymax]))
                if(classes[0][i] == 1):  # basketball
                    skz_glow_circle(img, (xCoor, yCoor), 25, SKZ_ORANGE, -1)
                    skz_pill(img, "BALL", (xCoor - 30, yCoor - 38), font_scale=0.7, accent=SKZ_ORANGE)
                    print("add basketball")
                    response.append({
                        'class': 'Basketball',
                        'detection_detail': {
                            'confidence': float("{:.5f}".format(scores[0][i])),
                            'center_coordinate': {'x': xCoor, 'y': yCoor},
                            'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                        }
                    })
                if(classes[0][i] == 2):  # Rim
                    cv2.rectangle(img, (xmin, ymax), (xmax, ymin), SKZ_DARK, 12)
                    cv2.rectangle(img, (xmin, ymax), (xmax, ymin), SKZ_ORANGE, 4)
                    skz_pill(img, "HOOP", (xCoor - 30, yCoor - 50), font_scale=0.7, accent=SKZ_ORANGE)
                    print("add hoop")
                    response.append({
                        'class': 'Hoop',
                        'detection_detail': {
                            'confidence': float("{:.5f}".format(scores[0][i])),
                            'center_coordinate': {'x': xCoor, 'y': yCoor},
                            'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                        }
                    })
        
        if(valid_detections < 2):
            for i in range(2):
                response.append({
                    'class': 'Not Found',
                    'detection_detail': {
                        'confidence': 0.0,
                        'center_coordinate': {'x': 0, 'y': 0},
                        'box_boundary': {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0}
                    }
                })
            
    return img

def detect_API(response, img):
    height, width = img.shape[:2]
    detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.36

    with tf.Session(graph=detection_graph, config=config) as sess:
        img_expanded = np.expand_dims(img, axis=0)
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: img_expanded})

        for i, box in enumerate(boxes[0]):
            if (scores[0][i] > 0.2):
                ymin = int((box[0] * height))
                xmin = int((box[1] * width))
                ymax = int((box[2] * height))
                xmax = int((box[3] * width))
                xCoor = int(np.mean([xmin, xmax]))
                yCoor = int(np.mean([ymin, ymax]))
                if(classes[0][i] == 1):  # basketball
                    response.append({
                        'class': 'Basketball',
                        'detection_detail': {
                            'confidence': float(scores[0][i]),
                            'center_coordinate': {'x': xCoor, 'y': yCoor},
                            'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                        }
                    })
                if(classes[0][i] == 2):  # Rim
                    response.append({
                        'class': 'Hoop',
                        'detection_detail': {
                            'confidence': float(scores[0][i]),
                            'center_coordinate': {'x': xCoor, 'y': yCoor},
                            'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                        }
                    })


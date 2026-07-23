import time
from absl import app, logging
import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
from .config import shooting_result
import sys
from sys import platform
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
    h = frame.shape[0]
    s = max(h / 720.0, 0.4)
    fs = font_scale * s
    thick = max(1, int(thickness * s))
    
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, SKZ_FONT, fs, thick)
    pad_x, pad_y = int(10 * s), int(6 * s)
    bar_w = int(4 * s)

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
                SKZ_FONT, fs, txt_color, thick, cv2.LINE_AA)


def skz_hud(frame, made, attempts, opaque=False):
    """Render a top-left scoreboard showing live Skillzo branding + score."""
    h, w = frame.shape[:2]
    s = max(h / 720.0, 0.4)
    bx, by, bw, bh = int(12 * s), int(12 * s), int(210 * s), int(74 * s)

    if opaque:
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), SKZ_DARK, -1)
    else:
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), SKZ_DARK, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Top orange stripe
    cv2.rectangle(frame, (bx, by), (bx + bw, by + max(1, int(4 * s))), SKZ_ORANGE, -1)

    # Brand name
    cv2.putText(frame, "SKILLZO AI", (bx + int(10 * s), by + int(22 * s)),
                SKZ_FONT, 0.5 * s, SKZ_ORANGE, max(1, int(1 * s)), cv2.LINE_AA)

    # Score
    score_str = f"{made}/{attempts}"
    cv2.putText(frame, score_str, (bx + int(10 * s), by + int(56 * s)),
                SKZ_FONT, 1.3 * s, SKZ_WHITE, max(1, int(2 * s)), cv2.LINE_AA)
    # Estimate width of score text dynamically for the 'MADE' placement
    (sw, _), _ = cv2.getTextSize(score_str, SKZ_FONT, 1.3 * s, max(1, int(2 * s)))
    cv2.putText(frame, "MADE", (bx + int(10 * s) + sw + int(8 * s), by + int(56 * s)),
                SKZ_FONT, 0.45 * s, (180, 180, 180), max(1, int(1 * s)), cv2.LINE_AA)


def skz_glow_circle(frame, center, radius, color, thickness=-1):
    """Draw a circle with a subtle glow ring around it."""
    h = frame.shape[0]
    s = max(h / 720.0, 0.4)
    r = int(radius * s)
    glow_color = tuple(min(int(c * 1.4), 255) for c in color)
    cv2.circle(frame, center, r + int(5 * s), glow_color, max(1, int(2 * s)), cv2.LINE_AA)
    t = thickness if thickness == -1 else max(1, int(thickness * s))
    cv2.circle(frame, center, r, color, t, cv2.LINE_AA)
    if thickness == -1:
        cv2.circle(frame, center, max(r - int(4 * s), 2), SKZ_WHITE, max(1, int(2 * s)), cv2.LINE_AA)


def skz_judgement(frame, text, center_x, center_y):
    """Large animated-style SCORE / MISS judgement overlay."""
    h = frame.shape[0]
    s = max(h / 720.0, 0.4)
    color  = SKZ_GREEN if text == "SCORE" else SKZ_RED
    label  = "● " + text
    scale  = 2.2 * s
    thick  = max(1, int(5 * s))
    (tw, th), _ = cv2.getTextSize(label, SKZ_FONT, scale, thick)
    tx = max(center_x - tw // 2, 4)
    ty = max(center_y - int(80 * s), th + 4)

    # Shadow
    cv2.putText(frame, label, (tx + int(3 * s), ty + int(3 * s)),
                SKZ_FONT, scale, SKZ_BLACK, thick + int(2 * s), cv2.LINE_AA)
    # Main text
    cv2.putText(frame, label, (tx, ty),
                SKZ_FONT, scale, color, thick, cv2.LINE_AA)

_cached_yolo_model = None
_cached_op_wrapper = None
_cached_op_datum = None

def yolo_init():
    global _cached_yolo_model
    if _cached_yolo_model is not None:
        return _cached_yolo_model
    
    model_path = os.path.join(os.getcwd(), 'best.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best.pt')
    
    _cached_yolo_model = YOLO(model_path)
    return _cached_yolo_model

def openpose_init():
    global _cached_op_wrapper, _cached_op_datum
    if _cached_op_wrapper is not None:
        return _cached_op_datum, _cached_op_wrapper
    try:
        if platform == "win32":
            sys.path.append(os.path.dirname(os.getcwd()))
            import OpenPose.Release.pyopenpose as op
        else:
            # In Modal/Ubuntu, pyopenpose is built directly in /openpose/build/python/openpose
            path = '/openpose/build/python/openpose'
            print("Importing pyopenpose from:", path)
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

def detect_shot(frame, trace, width, height, yolo_model, previous, during_shooting, shot_result, fig, datum, opWrapper, shooting_pose):
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

    # main YOLO detection
    results = yolo_model(frame, verbose=False)[0]
    boxes_list = []
    scores_list = []
    classes_list = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        boxes_list.append([y1/height, x1/width, y2/height, x2/width])
        scores_list.append(conf)
        classes_list.append(cls)
    boxes = [boxes_list]
    scores = [scores_list]
    classes = [classes_list]

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

    # Find the largest hoop index first
    best_hoop_idx = -1
    max_hoop_area = 0
    for i, box in enumerate(boxes[0]):
        if scores[0][i] > 0.2 and classes[0][i] == 2:
            h_ymin = int((box[0] * height))
            h_xmin = int((box[1] * width))
            h_ymax = int((box[2] * height))
            h_xmax = int((box[3] * width))
            h_area = (h_xmax - h_xmin) * (h_ymax - h_ymin)
            if h_area > max_hoop_area:
                max_hoop_area = h_area
                best_hoop_idx = i

    for i, box in enumerate(boxes[0]):
        if (scores[0][i] > 0.2):
            ymin = int((box[0] * height))
            xmin = int((box[1] * width))
            ymax = int((box[2] * height))
            xmax = int((box[3] * width))
            xCoor = int(np.mean([xmin, xmax]))
            yCoor = int(np.mean([ymin, ymax]))
            # Basketball (not head)
            if(classes[0][i] == 0 and (distance([headX, headY], [xCoor, yCoor]) > 30)):

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
                elif(ymin >= (previous['hoop_height'] - 30) and (distance([xCoor, yCoor], previous['ball']) < 250)):
                    # the moment when ball go below basket
                    if(during_shooting['isShooting']):
                        hoop_width = previous['hoop'][2] - previous['hoop'][0]
                        margin = hoop_width * 0.45 # 45% margin of error on each side
                        if(xCoor >= (previous['hoop'][0] - margin) and xCoor <= (previous['hoop'][2] + margin)):  # shot
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
                            shot_img = frame.copy()
                            skz_hud(shot_img, shooting_result.get('made', 0), shooting_result.get('attempts', 0), opaque=True)
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
                            shot_img = frame.copy()
                            skz_hud(shot_img, shooting_result.get('made', 0), shooting_result.get('attempts', 0), opaque=True)
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

            if(classes[0][i] == 2 and i == best_hoop_idx):  # Rim
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
    yolo_model = yolo_init()

    results = yolo_model(img, verbose=False)[0]
    boxes_list = []
    scores_list = []
    classes_list = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        boxes_list.append([y1/height, x1/width, y2/height, x2/width])
        scores_list.append(conf)
        classes_list.append(cls)
    boxes = [boxes_list]
    scores = [scores_list]
    classes = [classes_list]
    valid_detections = 0

    # Find the largest hoop index first
    best_hoop_idx = -1
    max_hoop_area = 0
    for i, box in enumerate(boxes[0]):
        if scores[0][i] > 0.2 and classes[0][i] == 2:
            h_ymin = int((box[0] * height))
            h_xmin = int((box[1] * width))
            h_ymax = int((box[2] * height))
            h_xmax = int((box[3] * width))
            h_area = (h_xmax - h_xmin) * (h_ymax - h_ymin)
            if h_area > max_hoop_area:
                max_hoop_area = h_area
                best_hoop_idx = i

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
            if(classes[0][i] == 0):  # basketball
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
            if(classes[0][i] == 2 and i == best_hoop_idx):  # Rim
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
    yolo_model = yolo_init()

    results = yolo_model(img, verbose=False)[0]
    boxes_list = []
    scores_list = []
    classes_list = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        boxes_list.append([y1/height, x1/width, y2/height, x2/width])
        scores_list.append(conf)
        classes_list.append(cls)
    boxes = [boxes_list]
    scores = [scores_list]
    classes = [classes_list]

    # Find the largest hoop index first
    best_hoop_idx = -1
    max_hoop_area = 0
    for i, box in enumerate(boxes[0]):
        if scores[0][i] > 0.2 and classes[0][i] == 2:
            h_ymin = int((box[0] * height))
            h_xmin = int((box[1] * width))
            h_ymax = int((box[2] * height))
            h_xmax = int((box[3] * width))
            h_area = (h_xmax - h_xmin) * (h_ymax - h_ymin)
            if h_area > max_hoop_area:
                max_hoop_area = h_area
                best_hoop_idx = i

    for i, box in enumerate(boxes[0]):
        if (scores[0][i] > 0.2):
            ymin = int((box[0] * height))
            xmin = int((box[1] * width))
            ymax = int((box[2] * height))
            xmax = int((box[3] * width))
            xCoor = int(np.mean([xmin, xmax]))
            yCoor = int(np.mean([ymin, ymax]))
            if(classes[0][i] == 0):  # basketball
                response.append({
                    'class': 'Basketball',
                    'detection_detail': {
                        'confidence': float(scores[0][i]),
                        'center_coordinate': {'x': xCoor, 'y': yCoor},
                        'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                    }
                })
            if(classes[0][i] == 2 and i == best_hoop_idx):  # Rim
                response.append({
                    'class': 'Hoop',
                    'detection_detail': {
                        'confidence': float(scores[0][i]),
                        'center_coordinate': {'x': xCoor, 'y': yCoor},
                        'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                    }
                })


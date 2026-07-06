# AI Basketball Analysis — Complete Setup & Integration Guide

> **Project:** Skillzo_Model (fork of [chonyy/AI-basketball-analysis](https://github.com/chonyy/AI-basketball-analysis))
> **Date:** July 5, 2026
> **Environment:** Google Colab (GPU backend) ↔ Windows 11 (local Flask frontend)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Package Versions & Requirements](#2-package-versions--requirements)
3. [Bugs Encountered & Fixes (Chronological)](#3-bugs-encountered--fixes-chronological)
4. [Feature Enhancements Added](#4-feature-enhancements-added)
5. [Complete Colab Setup Script](#5-complete-colab-setup-script)
6. [Local Windows Setup](#6-local-windows-setup)
7. [API Reference](#7-api-reference)
8. [Model Capabilities & Limitations](#8-model-capabilities--limitations)
9. [Skillzo Website Integration Blueprint](#9-skillzo-website-integration-blueprint)

---

## 1. Architecture Overview

The system uses a **split architecture** where the heavy GPU computation runs on Google Colab and the user-facing web UI runs on a local Windows machine.

```
┌──────────────────────┐         Cloudflare Tunnel         ┌──────────────────────────┐
│   Windows (Local)    │  ←──── .trycloudflare.com ────→   │   Google Colab (GPU)     │
│                      │                                    │                          │
│  app.py (Flask)      │   POST /api/upload_video           │  server.py (Flask)       │
│  127.0.0.1:5000      │   GET  /api/video_feed             │  0.0.0.0:5000            │
│                      │   GET  /api/shooting_result        │                          │
│  Templates:          │   GET  /api/detections/*.jpg        │  src/app_helper.py       │
│   - index.html       │   GET  /api/download_processed_video│  src/utils.py            │
│   - shooting_analysis│                                    │  src/config.py           │
│   - result.html      │                                    │                          │
│   - shot_detection   │                                    │  OpenPose (C++ bindings)  │
│                      │                                    │  TensorFlow (Object Det.) │
└──────────────────────┘                                    └──────────────────────────┘
```

**Data flow:**
1. User uploads a video on the Windows UI (`127.0.0.1:5000`)
2. `app.py` forwards the video to the Colab GPU server via Cloudflare Tunnel
3. Colab processes each frame using OpenPose (skeleton) + TensorFlow (ball detection)
4. Processed frames are streamed back live (MJPEG) for real-time viewing
5. After processing, stats (JSON) and graphs (JPG) are available via API
6. Result page displays stats, graphs, trajectory fitting, and analyzed video

---

## 2. Package Versions & Requirements

### Google Colab (GPU Server) — Python 3.10

| Package | Version | Purpose |
|---|---|---|
| **Python** | **3.10** (NOT 3.12) | Required for OpenPose C++ bindings compatibility |
| tensorflow | 2.15.0 | Object detection (basketball + hoop) |
| protobuf | 4.25.3 | TensorFlow serialization dependency |
| flask | latest | Web server for API endpoints |
| werkzeug | latest | Flask dependency |
| requests | latest | HTTP utilities |
| opencv-python-headless | latest | Video frame processing, VideoWriter |
| matplotlib | latest | Trajectory graph generation |
| scipy | latest | Curve fitting for trajectory parabola |
| pandas | latest | Data handling |
| filterpy | latest | Kalman filtering |
| imutils | latest | Image processing utilities |
| OpenPose | built from source | Human pose estimation (skeleton tracking) |
| pybind11 | bundled with OpenPose | Python ↔ C++ bridge |
| cloudflared | latest | Cloudflare Tunnel for public URL |
| ffmpeg | system package | Video codec conversion (mp4v → H.264) |

### Windows (Local Frontend) — Python 3.10 (conda env: `skillzo_env`)

| Package | Version | Purpose |
|---|---|---|
| flask | latest | Local web server |
| werkzeug | latest | Flask dependency |
| requests | latest | HTTP calls to Colab backend |

### Critical Version Constraint

> **⚠️ CAUTION:** Python 3.10 is mandatory on Colab. Google Colab defaults to Python 3.12, but OpenPose's `pybind11` bindings use the `f_code` C-API field which was removed in Python 3.11+. Using Python 3.12 will silently fail to compile the Python wrapper (`pyopenpose.so`).

---

## 3. Bugs Encountered & Fixes (Chronological)

### Bug 1: OpenPose Python Wrapper Fails to Compile (Python 3.12)

| | |
|---|---|
| **Error** | `ModuleNotFoundError: No module named 'pyopenpose'` / C++ compiler error mentioning `frame->f_code` |
| **Root Cause** | Colab's default Python 3.12 removed the `f_code` field from the Python C-API. The ancient `pybind11` bundled with OpenPose depends on it. The compiler silently skips building the Python bindings. |
| **Fix** | Install Python 3.10 and explicitly point CMake to 3.10's headers and libraries: |

```bash
echo "4. Compiling OpenPose bindings for Python 3.10 (This MUST take ~5 minutes!)..."
mkdir -p openpose/build
cd openpose/build
cmake -DBUILD_PYTHON=ON \
      -DPYTHON_EXECUTABLE=/usr/bin/python3.10 \
      -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so \
      -DPYTHON_INCLUDE_DIR=/usr/include/python3.10 \
      -DDOWNLOAD_BODY_25_MODEL=OFF \
      -DDOWNLOAD_BODY_COCO_MODEL=OFF \
      -DDOWNLOAD_BODY_MPI_MODEL=OFF \
      -DDOWNLOAD_FACE_MODEL=OFF \
      -DDOWNLOAD_HAND_MODEL=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_DOCS=OFF ..
make -j`nproc`
cd ../..
```

---

### Bug 2: CMake Caching Broken Builds

| | |
|---|---|
| **Error** | Recompiling OpenPose for Python 3.10 fails instantly, reuses broken files from the 3.12 attempt |
| **Root Cause** | `make` uses cached object files and the CMake cache from the previous failed build |
| **Fix** | Wipe the build artifacts before re-running CMake: |

```bash
rm -rf /content/openpose/build/python
rm -f /content/openpose/build/CMakeCache.txt
```

---

### Bug 3: Pip `distutils` Conflict

| | |
|---|---|
| **Error** | `Cannot uninstall blinker 1.4... It is a distutils installed project` |
| **Root Cause** | `pip install flask` tries to uninstall a system-managed package (`blinker`) that was installed via `apt-get` and is protected |
| **Fix** | Use `--ignore-installed` flag to bypass the protected package: |

```bash
echo "2. Installing Python 3.10, Pip, and C++ Dependencies..."
apt-get update -y
apt-get install -y python3.10 python3.10-dev python3.10-distutils libpython3.10-dev \
    libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev \
    libleveldb-dev libsnappy-dev libhdf5-serial-dev libatlas-base-dev \
    libopencv-dev libboost-all-dev liblmdb-dev
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
python3.10 -m pip install flask werkzeug requests tensorflow==2.15.0 protobuf==4.25.3 opencv-python-headless matplotlib scipy pandas filterpy imutils || true
```

---

### Bug 4: Matplotlib Colab Backend Crash

| | |
|---|---|
| **Error** | `ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value for backend` |
| **Root Cause** | Google Colab sets a hidden `MPLBACKEND` environment variable for inline notebook rendering. When our Flask server (running on Python 3.10) inherits this env var, it tries to load the Colab-specific backend and crashes |
| **Fix** | Override the env var when launching the server: |

```bash
MPLBACKEND=Agg nohup python3.10 server.py > app.log 2>&1 &
```

---

### Bug 5: Cloudflare Tunnel Expiration (500/502/522 Errors)

| | |
|---|---|
| **Error** | `requests.exceptions.JSONDecodeError: Expecting value: line 1 column 1` on any API call from Windows |
| **Root Cause** | The temporary Cloudflare Tunnel (`.trycloudflare.com`) disconnected or timed out during the long debugging session. `requests.post()` hit a dead URL, receiving a Cloudflare HTML error page instead of JSON |
| **Fix** | Restart the tunnel daemon, retrieve the new URL, and update `KAGGLE_GPU_URL` in local `app.py`: |

```python
# On Colab:
os.system("pkill cloudflared")
os.system("nohup cloudflared tunnel --url http://127.0.0.1:5000 > cloudflare.log 2>&1 &")
# Then grep the new URL from cloudflare.log and update app.py on Windows
```

---

### Bug 6: OpenPose PyBind11 Type Conversion Error (`emplaceAndPop`)

| | |
|---|---|
| **Error** | `TypeError: emplaceAndPop(): incompatible function arguments... Invoked with: <WrapperPython>, [<Datum>]` |
| **Root Cause** | Modern `pybind11` (compiled with Python 3.10) strictly disables implicit conversion from Python lists (`[datum]`) to C++ `std::vector` for opaque types |
| **Fix** | Patch `src/utils.py` to use OpenPose's explicit vector wrapper: |

```python
# Before (broken):
opWrapper.emplaceAndPop([datum])

# After (fixed):
import pyopenpose as op
opWrapper.emplaceAndPop(op.VectorDatum([datum]))
```

---

### Bug 7: JSON Serialization of NumPy Types

| | |
|---|---|
| **Error** | `TypeError: Object of type float32 is not JSON serializable` when clicking Results |
| **Root Cause** | The trajectory calculations return `numpy.float32` objects. Flask's `jsonify()` only knows native Python types |
| **Fix** | Add a recursive converter in `server.py` that calls `.item()` on NumPy scalars: |

```python
import numpy as np

def convert_numpy(obj):
    if isinstance(obj, np.generic): return obj.item()
    elif isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_numpy(v) for v in obj]
    return obj

# In the API route:
return jsonify(convert_numpy(shooting_result))
```

---

### Bug 8: `cv2.VideoCapture` Does Not Allow Custom Attributes

| | |
|---|---|
| **Error** | `AttributeError: 'cv2.VideoCapture' object has no attribute 'out_writer' and no __dict__ for setting new attributes` |
| **Root Cause** | OpenCV's `VideoCapture` is a C++ class exposed to Python. It does not have a Python `__dict__`, so you cannot set arbitrary attributes like `cap.out_writer = ...` |
| **Fix** | Store the `VideoWriter` in the local `shot_result` dictionary instead: |

```python
# Before (broken):
cap.out_writer = cv2.VideoWriter(...)

# After (fixed):
shot_result['out_writer'] = cv2.VideoWriter(...)
```

---

### Bug 9: `VideoWriter` Object Breaks JSON Serialization

| | |
|---|---|
| **Error** | `requests.exceptions.JSONDecodeError: Expecting value: line 1 column 1` when clicking Results after video analysis |
| **Root Cause** | The `VideoWriter` object was initially stored in `shooting_result` (the dictionary that gets serialized to JSON). Flask tried to `jsonify()` the C++ VideoWriter object and crashed, returning an HTML error page instead of JSON |
| **Fix** | Store the `VideoWriter` in `shot_result` (a local, non-serialized dictionary) instead of `shooting_result` (the global JSON-exposed dictionary) |

---

### Bug 10: `import os` Inside Function Shadows Module-Level Import

| | |
|---|---|
| **Error** | `UnboundLocalError: local variable 'os' referenced before assignment` at line 96 in `app_helper.py` |
| **Root Cause** | A patching script injected `import os` deep inside the `getVideoStream()` function body (next to the `ffmpeg` call). In Python, placing any `import` inside a function causes Python to treat that name as a local variable for the **entire** function scope. When `os.path.join()` was called earlier in the same function (before the import line), Python threw `UnboundLocalError` |
| **Fix** | Remove the redundant `import os` from inside the function body. The module-level `import os` at the top of the file is sufficient |

---

### Bug 11: Flask Route Placed After `app.run()` Is Never Registered

| | |
|---|---|
| **Error** | `HTTP Error 404: Not Found` when accessing `/api/download_processed_video` |
| **Root Cause** | A patching script appended the new route definition at the very bottom of `server.py`, **after** the `app.run()` call. In Flask, `app.run()` blocks forever, so any code placed after it is never executed and the route is never registered |
| **Fix** | Move the route definition above `if __name__ == '__main__':` and add a retry-wait loop for the ffmpeg conversion to finish: |

```python
@app.route("/api/download_processed_video")
def download_processed_video():
    for _ in range(15):
        if os.path.exists('./static/detections/final_output.mp4'):
            break
        time.sleep(1)
    if not os.path.exists('./static/detections/final_output.mp4'):
        return "Video not finished processing", 404
    return send_file('./static/detections/final_output.mp4', mimetype='video/mp4')

if __name__ == '__main__':
    app.run(...)
```

---

## 4. Feature Enhancements Added

### 4a. Frame-by-Frame JSON Data Export

**What:** The original model only returned aggregate averages. We patched `utils.py` to capture the basketball's X/Y coordinates and the player's elbow/knee angles on every single frame where the ball is detected.

**Where:** `src/config.py` — added `'frame_data': []` to `shooting_result` dict.
`src/utils.py` — inside `detect_shot()`, appends to `shooting_result['frame_data']` on every basketball detection.

**API Response (after patch):**
```json
{
    "attempts": 2,
    "made": 1,
    "miss": 1,
    "avg_elbow_angle": 69.62,
    "avg_knee_angle": 141.04,
    "avg_release_angle": 34.18,
    "avg_ballInHand_time": 1.07,
    "frame_data": [
        {"ball_x": 588.0, "ball_y": 357.0, "elbow_angle": 132.66, "knee_angle": 157.85},
        {"ball_x": 579.0, "ball_y": 316.0, "elbow_angle": 143.62, "knee_angle": 125.11}
    ]
}
```

### 4b. Normal-Speed Analyzed Video Playback

**What:** The original model only offered a slow, real-time MJPEG stream. We added background video saving using `cv2.VideoWriter` + `ffmpeg` H.264 conversion so users can replay the analyzed video at normal speed after processing.

**Backend changes:**
- `src/app_helper.py`: Uses `cv2.VideoWriter` to save each analyzed frame to `raw_output.mp4`, then runs `ffmpeg -vcodec libx264` to convert it to browser-compatible `final_output.mp4`
- `server.py`: New endpoint `/api/download_processed_video` serves the finished video

**Frontend changes:**
- `templates/result.html`: Added an HTML5 `<video controls autoplay loop>` player that loads from the new API endpoint

---

## 5. Complete Colab Setup Script

# STEP 3: Patch OpenPose VectorDatum Bug in utils.py
# ============================================================
print("4. Patching OpenPose VectorDatum Bug...")
utils_path = "/content/AI-basketball-analysis/src/utils.py"
with open(utils_path, "r") as f:
    content = f.read()

def replacer(match):
    indent = match.group(1)
    return f"{indent}import pyopenpose as op\n{indent}opWrapper.emplaceAndPop(op.VectorDatum([datum]))"

content = re.sub(
    r"^([ \t]*)opWrapper\.emplaceAndPop\(\[datum\]\)",
    replacer, content, flags=re.MULTILINE
)
with open(utils_path, "w") as f:
    f.write(content)

# ============================================================
# STEP 4: Add frame-by-frame data capture to utils.py
# ============================================================
print("5. Adding frame-by-frame JSON tracking...")
target_str = "if(classes[0][i] == 1 and (distance([headX, headY], [xCoor, yCoor]) > 30)):"
inject_str = """if(classes[0][i] == 1 and (distance([headX, headY], [xCoor, yCoor]) > 30)):
                if 'frame_data' not in shooting_result:
                    shooting_result['frame_data'] = []
                shooting_result['frame_data'].append({
                    "ball_x": float(xCoor), "ball_y": float(yCoor),
                    "elbow_angle": float(elbowAngle), "knee_angle": float(kneeAngle)
                })
"""
if "shooting_result['frame_data'].append" not in content:
    content = content.replace(target_str, inject_str)
    with open(utils_path, "w") as f:
        f.write(content)

# ============================================================
# STEP 5: Add frame_data to config.py
# ============================================================
config_path = "/content/AI-basketball-analysis/src/config.py"
with open(config_path, "r") as f:
    config_data = f.read()
if "'frame_data': []" not in config_data:
    config_data = config_data.replace(
        "'avg_ballInHand_time': 0",
        "'avg_ballInHand_time': 0,\n    'frame_data': []"
    )
    with open(config_path, "w") as f:
        f.write(config_data)

# ============================================================
# STEP 6: Add VideoWriter to app_helper.py for normal-speed video
# ============================================================
print("6. Adding background video saver...")
helper_path = "/content/AI-basketball-analysis/src/app_helper.py"
with open(helper_path, "r") as f:
    helper_data = f.read()

target_write = "            detection = cv2.resize(detection, (0, 0), fx=0.83, fy=0.83)"
inject_write = """            detection = cv2.resize(detection, (0, 0), fx=0.83, fy=0.83)
            if 'out_writer' not in shot_result:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                shot_result['out_writer'] = cv2.VideoWriter(
                    './static/detections/raw_output.mp4', fourcc, fps / 4.0,
                    (detection.shape[1], detection.shape[0]))
            shot_result['out_writer'].write(detection)"""

if "shot_result['out_writer']" not in helper_data:
    helper_data = helper_data.replace(target_write, inject_write)

target_end = "cv2.imwrite(trace_path, trace)"
inject_end = """cv2.imwrite(trace_path, trace)
    if 'out_writer' in shot_result and shot_result['out_writer'] is not None:
        shot_result['out_writer'].release()
        os.system("ffmpeg -y -i ./static/detections/raw_output.mp4 "
                  "-vcodec libx264 ./static/detections/final_output.mp4")"""

if "ffmpeg -y" not in helper_data:
    helper_data = helper_data.replace(target_end, inject_end)

with open(helper_path, "w") as f:
    f.write(helper_data)

# ============================================================
# STEP 7: Patch Flask server.py (NumPy converter + video route)
# ============================================================
print("7. Patching Flask server.py...")
server_path = "/content/AI-basketball-analysis/server.py"
with open(server_path, "r") as f:
    server_data = f.read()

# Add NumPy JSON converter
converter = """
import numpy as np
def convert_numpy(obj):
    if isinstance(obj, np.generic): return obj.item()
    elif isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_numpy(v) for v in obj]
    return obj
"""
if "convert_numpy" not in server_data:
    server_data = converter + "\n" + server_data
    server_data = server_data.replace(
        "return jsonify(shooting_result)",
        "return jsonify(convert_numpy(shooting_result))"
    )

# Add video download route (BEFORE app.run)
if "/api/download_processed_video" not in server_data:
    video_route = """
from flask import send_file
import time

@app.route("/api/download_processed_video")
def download_processed_video():
    for _ in range(15):
        if os.path.exists('./static/detections/final_output.mp4'):
            break
        time.sleep(1)
    if not os.path.exists('./static/detections/final_output.mp4'):
        return "Video not finished processing", 404
    return send_file('./static/detections/final_output.mp4', mimetype='video/mp4')

if __name__ == '__main__':
"""
    server_data = server_data.replace("if __name__ == '__main__':", video_route)

with open(server_path, "w") as f:
    f.write(server_data)

# ============================================================
# STEP 8: Launch server + Cloudflare Tunnel
# ============================================================
print("8. Launching Server & Cloudflare Tunnel...")
os.system("fuser -k 5000/tcp")
os.system("pkill cloudflared")
os.system("cd /content/AI-basketball-analysis && "
          "MPLBACKEND=Agg nohup python3.10 server.py > app.log 2>&1 &")
os.system("nohup cloudflared tunnel --url http://127.0.0.1:5000 "
          "> cloudflare.log 2>&1 &")

time.sleep(8)
try:
    url = subprocess.check_output(
        r"grep -o 'https://[^[:space:]]*\.trycloudflare\.com' cloudflare.log | head -n 1",
        shell=True).decode('utf-8').strip()
    print(f"\n✅ SETUP COMPLETE! Copy this URL into your Windows app.py:\n{url}")
except:
    print("Could not retrieve Cloudflare URL. Run: !cat cloudflare.log")
```

---

## 6. Local Windows Setup

### Prerequisites
- Python 3.10 (via Miniconda, env name: `skillzo_env`)
- Flask, requests, werkzeug installed in the conda env

### Steps
1. Clone the repository
2. Activate the conda environment: `conda activate skillzo_env`
3. Update `KAGGLE_GPU_URL` in `app.py` (line 16) with the Cloudflare URL from Colab
4. Run the local server: `python app.py`
5. Open browser to `http://127.0.0.1:5000`

### Key Local File Changes Made
- **`app.py` line 16**: Updated `KAGGLE_GPU_URL` to the current Cloudflare tunnel URL
- **`templates/result.html`**: Added HTML5 `<video>` player for normal-speed analyzed video playback

---

## 7. API Reference

All endpoints are served from the Colab backend via the Cloudflare Tunnel URL.

| Method | Endpoint | Description | Response |
|---|---|---|---|
| POST | `/api/upload_video` | Upload a video for analysis | `{"video_path": "..."}` |
| GET | `/api/video_feed?video_path=...` | Live MJPEG stream of frame-by-frame analysis | `multipart/x-mixed-replace` |
| GET | `/api/shooting_result` | Full analysis results as JSON (including `frame_data`) | JSON object |
| POST | `/api/process_image` | Upload a single image for detection | JSON with detection response |
| GET | `/api/detections/<filename>` | Serve a generated detection image | JPEG image |
| GET | `/api/download_processed_video` | Download the analyzed video at normal speed | MP4 video (H.264) |

### Example: Fetching Results via JavaScript

```javascript
const COLAB_URL = "https://your-tunnel.trycloudflare.com";

fetch(`${COLAB_URL}/api/shooting_result`, {
    headers: { "ngrok-skip-browser-warning": "true" }
})
.then(r => r.json())
.then(data => {
    console.log("Attempts:", data.attempts);
    console.log("Made:", data.made);
    console.log("Miss:", data.miss);
    console.log("Avg Elbow Angle:", data.avg_elbow_angle);
    console.log("Avg Knee Angle:", data.avg_knee_angle);
    console.log("Avg Release Angle:", data.avg_release_angle);
    console.log("Release Time:", data.avg_ballInHand_time, "sec");

    // Frame-by-frame data for Chart.js:
    data.frame_data.forEach((frame, i) => {
        console.log(`Frame ${i}: Ball(${frame.ball_x}, ${frame.ball_y}) `
                   + `Elbow: ${frame.elbow_angle}° Knee: ${frame.knee_angle}°`);
    });
});
```

---

## 8. Model Capabilities & Limitations

### What It Provides

| Category | Metrics |
|---|---|
| **Shot Counting** | Total Attempts, Made Shots, Missed Shots |
| **Biomechanical Angles** | Average Elbow Angle, Knee Angle, Release Angle |
| **Timing** | Ball-in-Hand Time (release speed) |
| **Visual Outputs** | Skeleton overlay video, Ball trajectory trace (JPG), Trajectory fitting curve (JPG) |
| **Raw Frame Data** | Per-frame ball X/Y coordinates and joint angles (JSON) |

### Camera Requirements

| Requirement | Detail |
|---|---|
| **Best Angle** | Side profile — camera on the sideline, seeing the player's full body and the entire arc to the basket |
| **Resolution** | Any standard resolution (the model resizes internally) |
| **NOT Supported** | Head-on angles (filming from under the hoop or behind the player) — flattens the arc and hides angles |

### Gameplay Limitations

- **Designed for isolated shooting drills** (free throws, catch-and-shoot)
- **Will break during 5v5 gameplay** — other players block the skeleton tracker and confuse ball detection
- **Single player only** — the model tracks `datum.poseKeypoints[0]` (the first detected person)

---

## 9. Skillzo Website Integration Blueprint

For a production Skillzo website, replace the temporary Colab + Cloudflare setup with a proper cloud pipeline:

1. **User Upload (Frontend)** → User uploads MP4 on Skillzo.com
2. **Cloud Storage (Backend)** → Backend uploads raw video to Amazon S3 / Google Cloud Storage
3. **GPU Worker (Colab Replacement)** → A dedicated GPU server (AWS, RunPod, or Modal) processes the video
4. **Processing** → GPU runs OpenPose + TensorFlow, generates JSON metrics + analyzed MP4
5. **Webhook** → GPU server sends a webhook to the main backend: "Done! Here's the JSON and video URL"
6. **Dashboard** → User's dashboard updates with stats, interactive Chart.js graphs using `frame_data`, and the analyzed video player

### Frontend Graph Reconstruction

Instead of displaying static JPG graphs, use the `frame_data` array from the JSON API to draw interactive charts with **Chart.js** or **Recharts**:

```javascript
// Extract ball coordinates for trajectory chart
const ballX = data.frame_data.map(f => f.ball_x);
const ballY = data.frame_data.map(f => f.ball_y);

new Chart(ctx, {
    type: 'scatter',
    data: {
        datasets: [{
            label: 'Ball Trajectory',
            data: ballX.map((x, i) => ({ x: x, y: ballY[i] })),
            pointBackgroundColor: 'rgba(82, 168, 50, 0.8)',
        }]
    }
});
```

---

> **NOTE:** This document was generated on July 5, 2026 during the initial setup and debugging session. If the Colab environment, OpenPose, or TensorFlow versions change in the future, some fixes may need to be adapted.

# Skillzo AI — Complete Developer Setup Guide

> **Last Updated:** July 6, 2026
> **Author:** Lokesh ([@Loki3306](https://github.com/Loki3306))

This is the **single source of truth** for setting up the entire Skillzo AI basketball analysis system from scratch. Follow it top-to-bottom and you will have a fully working local dev environment connected to a remote Kaggle GPU server.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Repository Setup (Clone Everything)](#3-repository-setup)
4. [Kaggle GPU Server Setup (Remote)](#4-kaggle-gpu-server-setup)
5. [Backend Setup (Local)](#5-backend-setup-local)
6. [Frontend Setup (Local)](#6-frontend-setup-local)
7. [Skillzo Model Local Helper (Local)](#7-skillzo-model-local-helper)
8. [Connect Everything](#8-connect-everything)
9. [Running the System (3 Terminals)](#9-running-the-system)
10. [API Reference](#10-api-reference)
11. [Troubleshooting](#11-troubleshooting)
12. [Model Capabilities & Limitations](#12-model-capabilities--limitations)

---

## 1. Architecture Overview

The system uses a **split architecture** with 3 local services and 1 remote GPU server:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LOCAL MACHINE (Windows)                            │
│                                                                            │
│   ┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────┐  │
│   │  FRONTEND (React) │   │  BACKEND (FastAPI)    │   │ MODEL (Flask)    │  │
│   │  localhost:5173   │──▶│  localhost:8000        │   │ localhost:5000   │  │
│   │  Vite dev server  │   │  Proxies GPU calls    │   │ Local helper     │  │
│   └──────────────────┘   └──────────┬─────────────┘   └──────────────────┘  │
│                                     │                                       │
└─────────────────────────────────────┼───────────────────────────────────────┘
                                      │  HTTPS (Pinggy Tunnel)
                                      ▼
                          ┌───────────────────────────┐
                          │  KAGGLE GPU SERVER (T4x2)  │
                          │  Python 3.10 + Flask       │
                          │  OpenPose + TensorFlow     │
                          │  Async processing          │
                          └───────────────────────────┘
```

**Data flow when a user uploads a video:**
1. User uploads video on the React frontend (`localhost:5173`)
2. Frontend uploads raw video to Supabase Storage, then calls backend
3. Backend downloads video from Supabase and uploads it to Kaggle via Pinggy tunnel
4. Kaggle starts async GPU processing (OpenPose skeleton + TensorFlow ball detection)
5. Backend polls Kaggle every 5 seconds until processing is done
6. Backend fetches JSON stats + processed MP4 from Kaggle
7. Backend uploads processed video to Supabase and returns stats to frontend
8. Frontend displays Shot Tracker with trajectory overlay and per-shot stats

---

## 2. Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| **Node.js** | 18+ | Frontend build tooling |
| **Python** | 3.10+ | Backend + local model helper |
| **Git** | Any | Cloning repos |
| **Kaggle Account** | Free tier | Remote GPU for AI processing |
| **Supabase Project** | Free tier | Auth, database, file storage |

### Supabase Setup
You need a Supabase project with:
- **Auth** enabled (email/password sign-up)
- **Storage buckets:** `videos` (for raw uploads) and `analysis` (for processed output)
- **Database table:** `videos` with columns matching the schema used by the backend

> Get your `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and `SUPABASE_SERVICE_ROLE_KEY` from your Supabase dashboard → Settings → API.

---

## 3. Repository Setup

Clone all three repositories:

```bash
# 1. The AI Model (Lokesh's fork with all OpenPose fixes baked in)
git clone https://github.com/Loki3306/Skillzo_Model.git

# 2. The Backend (FastAPI)
git clone https://github.com/skillzo-ai/skillzo-backend.git

# 3. The Frontend (React + Vite)
git clone https://github.com/skillzo-ai/skillzo-frontend.git
```

---

## 4. Kaggle GPU Server Setup

This is the heavy AI processing server that runs on Kaggle's free T4 GPUs. You need to run **two cells** in a Kaggle notebook every time you start a new session.

### 4.1 Create a Kaggle Notebook
1. Go to [kaggle.com](https://www.kaggle.com) → New Notebook
2. **Settings (right sidebar) → Accelerator → GPU T4 x2**
3. Make sure **Internet** is enabled

### 4.2 Cell 1: Initial Build & Installation (Run ONCE per new Kaggle session)

This installs Python 3.10, builds OpenPose from source (takes ~5-8 minutes), clones the Skillzo_Model repo, and installs all Python dependencies.

> ⚠️ **CRITICAL:** This cell takes 5-8 minutes because it compiles OpenPose from C++ source. Do NOT interrupt it. If it fails, re-run it from scratch.

```bash
%cd /kaggle/working

# 1. Install Python 3.10 and C++ build dependencies
!apt-get update -y
!apt-get install -y python3.10 python3.10-dev python3.10-distutils libpython3.10-dev \
    libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev \
    libleveldb-dev libsnappy-dev libhdf5-serial-dev libatlas-base-dev \
    libopencv-dev libboost-all-dev liblmdb-dev

# 2. Build OpenPose from source (MUST target Python 3.10, NOT 3.12!)
!rm -rf /kaggle/working/openpose
!git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git /kaggle/working/openpose
!mkdir -p /kaggle/working/openpose/build
%cd /kaggle/working/openpose/build
!cmake -DBUILD_PYTHON=ON \
    -DPYTHON_EXECUTABLE=/usr/bin/python3.10 \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so \
    -DPYTHON_INCLUDE_DIR=/usr/include/python3.10 \
    -DDOWNLOAD_BODY_25_MODEL=OFF \
    -DDOWNLOAD_BODY_COCO_MODEL=OFF \
    -DDOWNLOAD_BODY_MPI_MODEL=OFF \
    -DDOWNLOAD_FACE_MODEL=OFF \
    -DDOWNLOAD_HAND_MODEL=OFF ..
!make -j`nproc`

# 3. Clone the Skillzo Model repository
%cd /kaggle/working
!rm -rf /kaggle/working/Skillzo_Model
!git clone https://github.com/Loki3306/Skillzo_Model.git /kaggle/working/Skillzo_Model

# 4. Copy compiled OpenPose bindings into the model directory
!cp -r /kaggle/working/openpose/build/python/openpose /kaggle/working/Skillzo_Model/OpenPose/

# 5. Install Python packages for Python 3.10
!curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
!python3.10 -m pip install --ignore-installed numpy flask werkzeug requests \
    tensorflow==2.15.0 protobuf==4.25.3 opencv-python-headless \
    matplotlib scipy pandas filterpy imutils
```

### 4.3 Cell 2: Start Server & Tunnel (Run every time you need to restart)

This cell applies runtime patches, starts the Flask server, and opens a public tunnel via Pinggy so your local backend can reach it.

```python
import os
import time
import subprocess

# --- Patch 1: Restrict OpenPose to 1 GPU (prevents CUDA deadlock with TensorFlow) ---
print("1. Patching OpenPose GPU Limits...")
with open('/kaggle/working/Skillzo_Model/src/utils.py', 'r') as f:
    content = f.read()
if 'params["num_gpu"] = 1' not in content:
    content = content.replace(
        'params["model_folder"] = "./OpenPose/models"',
        'params["model_folder"] = "./OpenPose/models"\n    params["num_gpu"] = 1\n    params["num_gpu_start"] = 0'
    )
    with open('/kaggle/working/Skillzo_Model/src/utils.py', 'w') as f:
        f.write(content)

# --- Patch 2: Kill any leftover processes ---
print("2. Terminating existing processes...")
os.system("pkill -9 ssh")
os.system("fuser -k 5000/tcp")

# --- Start the Flask server on Python 3.10 ---
print("3. Restarting Server (Python 3.10)...")
os.system("cd /kaggle/working/Skillzo_Model && MPLBACKEND=Agg nohup python3.10 server.py > app.log 2>&1 &")
time.sleep(3)

# --- Start Pinggy tunnel (free, no account needed) ---
print("4. Starting Pinggy Tunnel...")
os.system("nohup ssh -p 443 -R0:localhost:5000 -o StrictHostKeyChecking=no a.pinggy.io > pinggy.log 2>&1 &")
time.sleep(7)

print("\n✅ Server started! Run the next cell to get your public URL.")
```

### 4.4 Cell 3: Get Your Tunnel URL

```python
!cat pinggy.log
```

Look for the URL like `https://xxxxx-xx-xx-xxx-xxx.free.pinggy.net`. **Copy this URL** — you'll paste it into your backend `.env` file next.

> ⚠️ **Pinggy free tier tunnels expire after 60 minutes.** When the tunnel drops, just re-run Cell 2 and Cell 3 to get a fresh URL, then update your backend `.env`.

### 4.5 Pulling Updates on Kaggle

If you or a teammate pushes code changes to `Skillzo_Model`, pull them on Kaggle without rebuilding OpenPose:

```python
!cd /kaggle/working/Skillzo_Model && git pull origin master
```

Then re-run **Cell 2** to restart the server with the new code.

---

## 5. Backend Setup (Local)

### 5.1 Install Dependencies

```bash
cd skillzo-backend
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 5.2 Configure Environment Variables

Copy the example env file and fill in your credentials:

```bash
cp env.example .env
```

Edit `.env` with your real values:

```env
# Supabase (get these from your Supabase dashboard → Settings → API)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here
FRONTEND_URL=http://localhost:5173

# Celery (for local dev, use eager mode — no Redis needed)
CELERY_TASK_ALWAYS_EAGER=1

# Roboflow & Email (get from team lead)
ROBOFLOW_API=your_roboflow_key
GMAIL_PASSWORD=your_app_password
SENDER_EMAIL=your_email@gmail.com

# Local ML bypass (set to 1 to skip ML processing for UI-only dev work)
MOCK_ML=0

# Kaggle GPU URL (paste the Pinggy URL from step 4.4 here)
KAGGLE_GPU_URL=https://xxxxx.free.pinggy.net
```

> ⚠️ **IMPORTANT:** Every time you restart Kaggle/Pinggy and get a new tunnel URL, you must update `KAGGLE_GPU_URL` in this file AND restart the backend server.

### 5.3 Run the Backend

```bash
uvicorn main:app --reload
```

The backend runs on `http://localhost:8000`.

---

## 6. Frontend Setup (Local)

### 6.1 Install Dependencies

```bash
cd skillzo-frontend
npm install
```

### 6.2 Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Supabase (same URL and anon key as backend)
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your_anon_key_here

# API URL (keep as /api for local dev — Vite proxies to localhost:8000)
VITE_API_URL=/api
```

### 6.3 Run the Frontend

```bash
npm run dev
```

The frontend runs on `http://localhost:5173`.

---

## 7. Skillzo Model Local Helper

The local `app.py` runs a lightweight Flask server for local image detection features (not GPU-dependent).

### 7.1 Setup

```bash
cd Skillzo_Model

# Create a conda environment (recommended) or venv
conda create -n skillzo_env python=3.10 -y
conda activate skillzo_env

pip install flask werkzeug requests
```

### 7.2 Run

```bash
python app.py
```

The local helper runs on `http://localhost:5000`.

---

## 8. Connect Everything

After completing steps 4–7, make sure:

| Component | URL | Status |
|-----------|-----|--------|
| Frontend | `http://localhost:5173` | `npm run dev` running |
| Backend | `http://localhost:8000` | `uvicorn main:app --reload` running |
| Local Model | `http://localhost:5000` | `python app.py` running |
| Kaggle GPU | `https://xxxxx.free.pinggy.net` | Pinggy tunnel active |
| `KAGGLE_GPU_URL` in backend `.env` | matches the Pinggy URL above | ✅ |

---

## 9. Running the System (3 Terminals)

Open **3 separate terminal windows** and run one command in each:

### Terminal 1: Local Model Helper
```bash
cd Skillzo_Model
conda activate skillzo_env
python app.py
```

### Terminal 2: Backend (FastAPI)
```bash
cd skillzo-backend
venv\Scripts\activate
uvicorn main:app --reload
```

### Terminal 3: Frontend (React/Vite)
```bash
cd skillzo-frontend
npm run dev
```

Then open `http://localhost:5173` in your browser. 🎉

---

## 10. API Reference

### Kaggle GPU Server Endpoints (via Pinggy tunnel)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload_video` | Upload a video file for processing |
| `GET` | `/api/start_processing?video_path=...` | Start async GPU processing (returns instantly) |
| `GET` | `/api/processing_status` | Poll processing state: `idle`, `running`, `done`, `error` |
| `GET` | `/api/shooting_result` | Fetch final analysis JSON (shots, angles, trajectories) |
| `GET` | `/api/download_processed_video` | Download the analyzed MP4 with skeleton overlays |
| `GET` | `/api/video_feed?video_path=...` | Legacy: live MJPEG stream (not used in current flow) |

### Backend Endpoints (localhost:8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/video-analysis/analyze-gpu` | Full pipeline: upload → process → fetch results → save to Supabase |

---

## 11. Troubleshooting

### OpenPose Fails to Compile on Kaggle
**Symptom:** `ModuleNotFoundError: No module named 'pyopenpose'`
**Cause:** Kaggle defaults to Python 3.12, which removed the `f_code` C-API field that OpenPose's `pybind11` depends on.
**Fix:** Always use `python3.10` and point CMake to `3.10` headers/libraries (already done in Cell 1). If it still fails, wipe the build cache:
```bash
!rm -rf /kaggle/working/openpose/build/python
!rm -f /kaggle/working/openpose/build/CMakeCache.txt
```
Then re-run Cell 1.

### OpenPose `emplaceAndPop` TypeError
**Symptom:** `TypeError: emplaceAndPop(): incompatible function arguments`
**Cause:** Modern `pybind11` disables implicit list → `std::vector` conversion.
**Fix:** Already patched in `src/utils.py` — uses `op.VectorDatum([datum])` instead of `[datum]`.

### GPU Deadlock / 15-Minute Processing Times
**Symptom:** Video processing hangs or takes 15+ minutes for a 5-second clip.
**Cause:** OpenPose greedily claims both GPUs, starving TensorFlow.
**Fix:** Cell 2 already injects `params["num_gpu"] = 1` to restrict OpenPose to one GPU.

### Pip `distutils` Conflict on Kaggle
**Symptom:** `Cannot uninstall blinker 1.4... It is a distutils installed project`
**Fix:** Already handled in Cell 1 with `--ignore-installed` flag.

### Matplotlib Backend Crash
**Symptom:** `ValueError: Key backend: 'module://matplotlib_inline.backend_inline'`
**Cause:** Kaggle injects `MPLBACKEND` env var for inline notebook rendering.
**Fix:** Cell 2 launches the server with `MPLBACKEND=Agg` to force headless mode.

### Pinggy Tunnel Drops / Connection Reset
**Symptom:** `ConnectionResetError` or `TimeoutError` from the backend.
**Cause:** Pinggy free tier tunnels expire after 60 minutes.
**Fix:** Re-run Kaggle Cell 2 and Cell 3. Copy the new URL into backend `.env` and restart `uvicorn`.

### Backend Still Using Old Pinggy URL After `.env` Update
**Symptom:** Error mentions old URL hostname even after editing `.env`.
**Cause:** `uvicorn --reload` watches code files, not `.env`.
**Fix:** `Ctrl+C` the backend and re-run `uvicorn main:app --reload`.

### NumPy JSON Serialization Error
**Symptom:** `TypeError: Object of type float32 is not JSON serializable`
**Fix:** Already handled in `server.py` with the `convert_numpy()` recursive converter.

### Large Video Upload Timeout
**Symptom:** Upload works for small videos but fails for 20MB+ files.
**Cause:** Pinggy's free tier bandwidth is limited (~1 MB/s).
**Fix:** Backend timeouts are set to 600 seconds (10 minutes) for upload/download operations.

---

## 12. Model Capabilities & Limitations

### What It Detects

| Category | Metrics |
|----------|---------|
| **Shot Counting** | Total attempts, made shots, missed shots |
| **Biomechanical Angles** | Elbow angle, knee angle, release angle (per shot) |
| **Timing** | Ball-in-hand time / release speed (per shot) |
| **Trajectory** | Ball X/Y coordinates per frame, hoop bounding box |
| **Visual Output** | Skeleton overlay video (MP4), trajectory trace |

### Camera Requirements

| Requirement | Detail |
|-------------|--------|
| **Best angle** | Side profile — camera on sideline, seeing full body + full arc to basket |
| **Resolution** | Any standard resolution (model resizes internally) |
| **NOT supported** | Head-on angles (under hoop or behind player) — flattens the arc |

### Limitations

- **Single player only** — tracks `datum.poseKeypoints[0]` (first detected person)
- **Designed for isolated shooting drills** (free throws, catch-and-shoot)
- **Will break during 5v5 gameplay** — other players confuse skeleton + ball detection
- **Requires clear ball visibility** — ball must be visible in frame for detection

---

> **Questions?** Reach out to the team or check the detailed historical debugging logs in `SETUP_GUIDE.md`, `TROUBLESHOOTING.md`, and `latest_setup_guide.md` inside the `Skillzo_Model` repository.

# AI Basketball Analysis - Latest Setup Guide & Architecture Fixes

This document serves as the definitive setup guide and post-mortem for running the AI Basketball Analysis model on remote GPU environments (like Kaggle or Colab) via tunneling services. It documents all the critical bug fixes, memory leaks, and deadlocks that were resolved to achieve a stable, real-time processing pipeline.

## 🚀 The Ultimate Kaggle Startup Script

To guarantee a flawless, deadlock-free environment on Kaggle, you must run this exact script in a single cell every time you spin up a new environment. 

This script automatically patches OpenPose GPU greediness, disables Cloudflare buffering, and explicitly forces Python 3.10 to prevent C-API crashes.

```python
import os
import time
import subprocess

print("1. Patching OpenPose GPU Limits...")
# Fixes the CUDA Deadlock by preventing OpenPose from stealing both GPUs
with open('/kaggle/working/Skillzo_Model/src/utils.py', 'r') as f:
    content = f.read()
if 'params["num_gpu"] = 1' not in content:
    content = content.replace(
        'params["model_folder"] = "./OpenPose/models"',
        'params["model_folder"] = "./OpenPose/models"\n    params["num_gpu"] = 1\n    params["num_gpu_start"] = 0'
    )
    with open('/kaggle/working/Skillzo_Model/src/utils.py', 'w') as f:
        f.write(content)

print("2. Patching Cloudflare MJPEG Buffering...")
# Injects headers to prevent Cloudflare from buffering the live video feed
with open('/kaggle/working/Skillzo_Model/server.py', 'r') as f:
    content = f.read()
if 'X-Accel-Buffering' not in content:
    content = content.replace(
        "mimetype='multipart/x-mixed-replace; boundary=frame'", 
        "mimetype='multipart/x-mixed-replace; boundary=frame', headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'}"
    )
    with open('/kaggle/working/Skillzo_Model/server.py', 'w') as f:
        f.write(content)

print("3. Downloading Cloudflare daemon...")
os.system("wget -q -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64")
os.system("chmod +x cloudflared")

print("4. Terminating existing processes...")
os.system("pkill -9 cloudflared")
os.system("fuser -k 5000/tcp")

print("5. Restarting Server (Python 3.10)...")
# Must use python3.10 specifically! Python 3.12 will crash pyopenpose.
os.system("cd /kaggle/working/Skillzo_Model && MPLBACKEND=Agg nohup python3.10 server.py > app.log 2>&1 &")
time.sleep(3)

print("6. Starting Cloudflare Tunnel...")
os.system("nohup ./cloudflared tunnel --url http://127.0.0.1:5000 > cloudflared.log 2>&1 &")
time.sleep(7)

try:
    url = subprocess.check_output(r"grep -o 'https://[^[:space:]]*\.trycloudflare\.com' cloudflared.log | head -n 1", shell=True).decode('utf-8').strip()
    print(f"\n✅ SUCCESS! Your Cloudflare URL is:\n{url}")
except:
    print("Could not retrieve Cloudflare URL. Please check cloudflared.log")
```

---

## 🛠️ Summary of Critical Fixes Applied to the Codebase

The local `app.py` and remote files were heavily modified to fix severe stability issues. Here is the historical log of what was fixed and why:

### 1. OpenPose vs. TensorFlow GPU Deadlocks (The 15-Minute Hang)
- **The Bug:** Kaggle's T4x2 environment provides two GPUs. By default, OpenPose initialized and aggressively claimed 100% of the memory on *both* GPUs. When TensorFlow subsequently tried to boot up to track the basketball, it either fell back to the CPU (causing a 3-second video to take 15 minutes to process) or caused a total CUDA deadlock, freezing the entire Python interpreter.
- **The Fix:** We injected `params["num_gpu"] = 1` into `src/utils.py`. This restricts OpenPose to exactly one GPU, leaving the second GPU entirely free for TensorFlow. Processing time dropped from 15+ minutes back to a few seconds.

### 2. TensorFlow Memory Leaks (The Singleton Pattern)
- **The Bug:** Originally, `app_helper.py` instantiated a brand new `tf.Graph()` and re-read the frozen model weights from disk into VRAM *every single time* a video or image was uploaded. This caused the Kaggle GPU to run out of memory (OOM) and crash after just 2 or 3 uploads.
- **The Fix:** We implemented the **Singleton Pattern** in `app_helper.py` using a global `detection_graph` variable. The heavy TensorFlow model is now loaded exactly once upon the first request and kept in memory for all subsequent videos, completely eliminating the memory leak.

### 3. Cloudflare Stream Buffering (The Blank Screen)
- **The Bug:** We migrated from Pinggy to Cloudflare Tunnels for better stability. However, Cloudflare's edge proxy aggressively buffers `multipart/x-mixed-replace` streams (MJPEG Server-Sent Events). This caused the live video player on the Windows browser to hang on a blank screen until the entire video was fully processed by Kaggle.
- **The Fix:** We patched the remote `server.py` to inject `X-Accel-Buffering: no` headers. More importantly, we updated the local Windows `app.py` to catch the resulting `requests.exceptions.Timeout` on the `/result` page. The local server now gracefully intercepts the timeout and renders a beautiful CSS "Processing..." page that auto-refreshes every 5 seconds until Kaggle unlocks the Python GIL and delivers the final MP4.

### 4. Browser "Confirm Form Resubmission" Errors
- **The Bug:** The user was getting annoying browser warnings when trying to hit the "Back" button from the results page to check progress.
- **The Fix:** Changed the HTML form method in `shooting_analysis.html` from `POST` to `GET`. The `/result` route in `app.py` was updated to support this, allowing flawless, warning-free backward and forward browser navigation.

### 5. Python 3.12 Incompatibility
- **The Bug:** Kaggle defaults to Python 3.12, but running `python server.py` caused an immediate `ModuleNotFoundError: No module named 'pyopenpose'`.
- **The Fix:** OpenPose's `pybind11` C-bindings rely on the `f_code` API which was completely removed in Python 3.11+. We must explicitly enforce the use of `python3.10` in all startup scripts to prevent total failure.

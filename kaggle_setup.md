# Kaggle Setup Guide

When you start a **brand new Kaggle session** (where the environment is completely wiped clean), you only ever need exactly **TWO code blocks**. 

Keep these safe in your Kaggle notebook!

---

### CELL 1: The Initial Build & Installation (Run once per new Kaggle session)
This block sets up Python 3.10, builds OpenPose from scratch, clones your repository, and installs the required Python packages. Because Kaggle wipes its memory when you shut it down, you must run this first whenever you start a new machine.

```bash
%cd /kaggle/working
!apt-get update -y
!apt-get install -y python3.10 python3.10-dev python3.10-distutils libpython3.10-dev libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev libatlas-base-dev libopencv-dev libboost-all-dev liblmdb-dev

# 1. Build OpenPose
!rm -rf /kaggle/working/openpose
!git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git /kaggle/working/openpose
!mkdir -p /kaggle/working/openpose/build
%cd /kaggle/working/openpose/build
!cmake -DBUILD_PYTHON=ON -DPYTHON_EXECUTABLE=/usr/bin/python3.10 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so -DPYTHON_INCLUDE_DIR=/usr/include/python3.10 -DDOWNLOAD_BODY_25_MODEL=OFF -DDOWNLOAD_BODY_COCO_MODEL=OFF -DDOWNLOAD_BODY_MPI_MODEL=OFF -DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF ..
!make -j`nproc`

# 2. Setup Your Repository
%cd /kaggle/working
!rm -rf /kaggle/working/Skillzo_Model
!git clone https://github.com/Loki3306/Skillzo_Model.git /kaggle/working/Skillzo_Model
!cp -r /kaggle/working/openpose/build/python/openpose /kaggle/working/Skillzo_Model/OpenPose/

# 3. Install Python Dependencies
!curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
!python3.10 -m pip install --ignore-installed numpy flask werkzeug requests tensorflow==2.15.0 protobuf==4.25.3 opencv-python-headless matplotlib scipy pandas filterpy imutils
```

---

### CELL 2: The Ultimate Startup Script (Run to start the server)
Once Cell 1 is finished, run this cell. It automatically applies all the GPU deadlock patches we figured out, disables Cloudflare buffering, and launches the server. (If your Kaggle kernel ever crashes but the files are still there, you only need to run this cell!)

```python
import os
import time
import subprocess

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

print("2. Patching Cloudflare MJPEG Buffering...")
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

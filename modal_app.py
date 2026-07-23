import modal
import os
import sys

# 1. Define the Container Environment
# We start with Debian, install all required C++ libraries, compile OpenPose, and pip install Python packages.
image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
    .apt_install(
        "git", "cmake", "make", "g++", "wget",
        "libprotobuf-dev", "protobuf-compiler", "libgoogle-glog-dev", 
        "libgflags-dev", "libleveldb-dev", "libsnappy-dev", 
        "libhdf5-serial-dev", "libatlas-base-dev", "libopencv-dev", 
        "libboost-all-dev", "liblmdb-dev"
    )
    .run_commands(
        # Clone OpenPose
        "git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git /openpose",
        "mkdir -p /openpose/build",
        # Run CMake to configure the build (matching your Kaggle settings exactly)
        "cd /openpose/build && cmake -DBUILD_PYTHON=ON "
        "-DPYTHON_EXECUTABLE=/usr/local/bin/python "
        "-DDOWNLOAD_BODY_25_MODEL=OFF "
        "-DDOWNLOAD_BODY_COCO_MODEL=OFF "
        "-DDOWNLOAD_BODY_MPI_MODEL=OFF "
        "-DDOWNLOAD_FACE_MODEL=OFF "
        "-DDOWNLOAD_HAND_MODEL=OFF ..",
        # Compile it using all available CPU cores
        "cd /openpose/build && make -j$(nproc)",
    )
    .pip_install(
        "numpy", "flask", "werkzeug", "requests", "ultralytics",
        "tensorflow==2.15.0", "protobuf==4.25.3", "opencv-python-headless", 
        "matplotlib", "scipy", "pandas", "filterpy", "imutils"
    )
    .apt_install("ffmpeg")
)

# 2. Create the Modal App
app = modal.App("skillzo-model-api")

# Mount your local repository files into the Modal container
# This ensures that when the container boots, it has access to `server.py`, `src/`, etc.
image = image.add_local_dir(".", remote_path="/root/Skillzo_Model")

# 3. Define the Web Endpoint
# We wrap your existing Flask app (server.py) using modal.wsgi_app()
# We request a T4 GPU, which is cheap but powerful enough for OpenPose.
@app.function(image=image, gpu="T4", timeout=3600)
@modal.wsgi_app()
def flask_app():
    # Change working directory so Flask paths (like ./static/uploads) work correctly
    os.chdir("/root/Skillzo_Model")
    
    # We will import pyopenpose directly from /openpose/build/python/openpose in utils.py
    # Patch utils.py to restrict GPU usage just like you did in Kaggle
    with open('/root/Skillzo_Model/src/utils.py', 'r') as f:
        content = f.read()
    if 'params["num_gpu"] = 1' not in content:
        content = content.replace(
            'params["model_folder"] = "./OpenPose/models"',
            'params["model_folder"] = "./OpenPose/models"\n    params["num_gpu"] = 1\n    params["num_gpu_start"] = 0'
        )
        with open('/root/Skillzo_Model/src/utils.py', 'w') as f:
            f.write(content)
            
    # Finally, import your Flask app!
    import sys
    if "/root/Skillzo_Model" not in sys.path:
        sys.path.insert(0, "/root/Skillzo_Model")
    from server import app as skillzo_flask_app
    return skillzo_flask_app

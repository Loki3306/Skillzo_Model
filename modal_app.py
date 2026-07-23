import modal
import os
import sys

# 1. Define the Container Environment
# We start with Debian, install all required C++ libraries, compile OpenPose, and pip install Python packages.
image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
    .apt_install("git", "wget", "ffmpeg", "libsm6", "libxext6")
    .pip_install(
        "numpy", "flask", "werkzeug", "requests", "ultralytics",
        "tensorflow==2.15.0", "protobuf==4.25.3", "opencv-python-headless", 
        "matplotlib", "scipy", "pandas", "filterpy", "imutils"
    )
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

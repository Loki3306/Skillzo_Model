# AI Basketball Analysis: Colab Setup & Troubleshooting Guide

During the setup of the AI Basketball Analysis tool on Google Colab, we encountered several advanced environment and compatibility issues, primarily due to Colab upgrading its default environment to Python 3.12 and strict compiler changes. 

Below is a complete record of every error encountered and its corresponding fix to ensure future deployments go smoothly.

## 1. OpenPose Python Wrapper Fails to Compile on Python 3.12
**Error:** `ModuleNotFoundError: No module named 'pyopenpose'` or C++ compiler error mentioning `frame->f_code` on line 2026.
**Cause:** Google Colab updated its default Python version to 3.12. The ancient `pybind11` library bundled with OpenPose uses Python C-API features (like `f_code`) that were completely removed in Python 3.11+. The C++ compiler silently skips building the Python bindings.
**Fix:** Bypass Python 3.12 entirely by installing Python 3.10 and explicitly directing CMake to use the 3.10 headers and libraries. 

## 2. CMake Caching Broken Builds
**Error:** Compiling OpenPose for Python 3.10 fails instantly, reusing broken files from the 3.12 attempt. 
**Cause:** `make` relies on cached object files and CMake cache.
**Fix:** Completely wipe the Python build directory and the `CMakeCache.txt` file before running the `cmake` configuration for Python 3.10.

## 3. Pip Installation "distutils" Conflict
**Error:** `Cannot uninstall blinker 1.4... It is a distutils installed project and thus we cannot accurately determine which files belong to it`
**Cause:** When attempting to install Flask in Python 3.10, pip tries to uninstall a protected system package (`blinker`) that was installed via `apt-get`.
**Fix:** Force pip to bypass the uninstallation by using the `--ignore-installed` flag: 
`python3.10 -m pip install --ignore-installed flask werkzeug requests tensorflow==2.15.0 protobuf==4.25.3 opencv-python-headless matplotlib scipy pandas filterpy imutils`

## 4. Matplotlib Colab Backend Crash
**Error:** `ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value for backend`
**Cause:** Google Colab injects a hidden environment variable (`MPLBACKEND`) into the session to draw graphs inline. Since the Flask server runs in the background on a custom Python 3.10 binary, it crashes attempting to load this Colab-specific backend.
**Fix:** Override the environment variable when launching the server to strictly use the headless `Agg` backend: 
`MPLBACKEND=Agg nohup python3.10 server.py > app.log 2>&1 &`

## 5. Cloudflare Tunnel Expiration (500/502/522 Errors on Windows)
**Error:** Windows local terminal shows `requests.exceptions.JSONDecodeError: Expecting value: line 1 column 1` on `/basketball_detection`.
**Cause:** The temporary Cloudflare Tunnel (`.trycloudflare.com`) disconnected or timed out during the long debugging session. `requests.post()` hit a dead URL, returning a Cloudflare HTML error page instead of JSON.
**Fix:** Restart the `cloudflared` tunnel daemon, retrieve the newly generated URL from the logs, and update `KAGGLE_GPU_URL` in the local `app.py`.

## 6. OpenPose PyBind11 Type Conversion Error
**Error:** `TypeError: emplaceAndPop(): incompatible function arguments... Invoked with: <pyopenpose.WrapperPython object>, [<pyopenpose.Datum object>]`
**Cause:** Modern versions of PyBind11 strictly disable implicit conversion from Python lists (`[]`) to C++ `std::vector` for opaque types. Passing `[datum]` directly to C++ throws a TypeError.
**Fix:** Patch `/content/AI-basketball-analysis/src/utils.py` to explicitly wrap the Python list in OpenPose's exposed vector wrapper: `op.VectorDatum([datum])`.

## 7. JSON Serialization of NumPy Types
**Error:** `TypeError: Object of type float32 is not JSON serializable` when clicking the Results page.
**Cause:** The AI trajectory tracing returns calculations as NumPy `float32` objects. Flask's strict `jsonify()` function only knows how to serialize native Python floats.
**Fix:** Inject a recursive `convert_numpy` dictionary parser into `server.py` that intercepts the `shooting_result` output and calls `.item()` on any `np.generic` types to convert them to native Python numbers before returning the JSON.

---

## Final Working Colab Setup Script

For future deployments, you can run this all-in-one Python script in a Colab cell to automatically perform all fixes, install dependencies, patch OpenPose, and start the server:

```python
import os
import re
import subprocess
import time

print("1. Installing Python 3.10 and Dependencies...")
os.system("apt-get install -y python3.10 python3.10-dev python3.10-distutils libpython3.10-dev > /dev/null")
os.system("curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 > /dev/null 2>&1")
os.system("python3.10 -m pip install --ignore-installed flask werkzeug requests tensorflow==2.15.0 protobuf==4.25.3 opencv-python-headless matplotlib scipy pandas filterpy imutils > /dev/null 2>&1")

print("2. Clearing old CMake cache...")
os.system("rm -rf /content/openpose/build/python")
os.system("rm -f /content/openpose/build/CMakeCache.txt")

print("3. Compiling OpenPose bindings for Python 3.10...")
cmake_cmd = "cmake -DBUILD_PYTHON=ON -DPYTHON_EXECUTABLE=/usr/bin/python3.10 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so -DPYTHON_INCLUDE_DIR=/usr/include/python3.10 .."
os.system(f"cd /content/openpose/build && {cmake_cmd} > /dev/null 2>&1")
os.system("cd /content/openpose/build && make -j`nproc` > /dev/null 2>&1")
os.system("cp /content/openpose/build/python/openpose/pyopenpose*.so /content/AI-basketball-analysis/pyopenpose.so")
os.system("cp /content/openpose/build/python/openpose/pyopenpose*.so /content/AI-basketball-analysis/src/pyopenpose.so")

print("4. Patching OpenPose VectorDatum Bug...")
utils_path = "/content/AI-basketball-analysis/src/utils.py"
with open(utils_path, "r") as f:
    content = f.read()
def replacer(match):
    indent = match.group(1)
    return f"{indent}import pyopenpose as op\\n{indent}opWrapper.emplaceAndPop(op.VectorDatum([datum]))"
content = re.sub(r"^([ \t]*)opWrapper\.emplaceAndPop\(\[datum\]\)", replacer, content, flags=re.MULTILINE)
with open(utils_path, "w") as f:
    f.write(content)

print("5. Patching Flask float32 JSON Serialization Bug...")
server_path = "/content/AI-basketball-analysis/server.py"
with open(server_path, "r") as f:
    content = f.read()
converter = \"\"\"
import numpy as np
def convert_numpy(obj):
    if isinstance(obj, np.generic): return obj.item()
    elif isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_numpy(v) for v in obj]
    return obj
\"\"\"
if "convert_numpy" not in content:
    content = converter + "\\n" + content
    content = content.replace("return jsonify(shooting_result)", "return jsonify(convert_numpy(shooting_result))")
    with open(server_path, "w") as f:
        f.write(content)

print("6. Launching Server & Cloudflare Tunnel...")
os.system("fuser -k 5000/tcp")
os.system("pkill cloudflared")
os.system("cd /content/AI-basketball-analysis && MPLBACKEND=Agg nohup python3.10 server.py > app.log 2>&1 &")
os.system("nohup cloudflared tunnel --url http://127.0.0.1:5000 > cloudflare.log 2>&1 &")

time.sleep(8)
try:
    url = subprocess.check_output(r"grep -o 'https://[^[:space:]]*\.trycloudflare\.com' cloudflare.log | head -n 1", shell=True).decode('utf-8').strip()
    print(f"\\n✅ SETUP COMPLETE! Copy this URL into your Windows app.py:\\n{url}")
except:
    print("Could not retrieve Cloudflare URL.")
```

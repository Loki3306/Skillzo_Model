import os, time, numpy as np, threading
from flask import Flask, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename
from src.app_helper import getVideoStream
from src.config import shooting_result

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
os.makedirs('./static/uploads', exist_ok=True)
os.makedirs('./static/detections', exist_ok=True)

# --- Async processing state ---
_processing_lock = threading.Lock()
_processing_status = {"state": "idle", "error": None}  # idle | running | done | error

def convert_numpy(obj):
    if isinstance(obj, np.generic): return obj.item()
    elif isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_numpy(v) for v in obj]
    return obj

@app.route('/api/upload_video', methods=['POST'])
def handle_video():
    f = request.files['video']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
    f.save(filepath)
    return jsonify({'video_path': filepath})

@app.route('/api/video_feed')
def video_feed():
    hoop_x = request.args.get('hoop_x', type=float, default=None)
    hoop_y = request.args.get('hoop_y', type=float, default=None)
    stream = getVideoStream(request.args.get('video_path'), hoop_x, hoop_y)
    return Response(stream, mimetype='multipart/x-mixed-replace; boundary=frame', headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})

# --- Async processing: start in background thread, poll for status ---

def _run_processing(video_path, hoop_x=None, hoop_y=None):
    """Background worker that runs the full AI pipeline."""
    global _processing_status
    try:
        stream = getVideoStream(video_path, hoop_x, hoop_y)
        for _ in stream:
            pass
        with _processing_lock:
            _processing_status = {"state": "done", "error": None}
    except Exception as e:
        with _processing_lock:
            _processing_status = {"state": "error", "error": str(e)}

@app.route('/api/start_processing')
def start_processing():
    """Kick off processing in a background thread. Returns instantly."""
    global _processing_status
    video_path = request.args.get('video_path')
    hoop_x = request.args.get('hoop_x', type=float, default=None)
    hoop_y = request.args.get('hoop_y', type=float, default=None)
    if not video_path:
        return jsonify({"error": "video_path is required"}), 400

    with _processing_lock:
        if _processing_status["state"] == "running":
            return jsonify({"status": "already_running"}), 409
        _processing_status = {"state": "running", "error": None}

    t = threading.Thread(target=_run_processing, args=(video_path, hoop_x, hoop_y), daemon=True)
    t.start()
    return jsonify({"status": "started"})

@app.route('/api/processing_status')
def processing_status():
    """Poll this to check if processing is done."""
    with _processing_lock:
        return jsonify(_processing_status)

# Legacy sync endpoint (kept for backwards compat)
@app.route('/api/process_video_fast')
def process_video_fast():
    hoop_x = request.args.get('hoop_x', type=float, default=None)
    hoop_y = request.args.get('hoop_y', type=float, default=None)
    stream = getVideoStream(request.args.get('video_path'), hoop_x, hoop_y)
    for _ in stream:
        pass
    return jsonify({"status": "done"})

@app.route('/api/shooting_result')
def get_shooting_result():
    return jsonify(convert_numpy(shooting_result))

@app.route("/api/download_processed_video")
def download_processed_video():
    if not os.path.exists('./static/detections/final_output.mp4'): return "Video not finished processing", 404
    return send_file('./static/detections/final_output.mp4', mimetype='video/mp4')

@app.route("/api/download_shot_image")
def download_shot_image():
    shot_id = request.args.get('shot_id')
    if not shot_id: return "shot_id is required", 400
    path = f'./static/detections/shots/shot_{shot_id}.jpg'
    if not os.path.exists(path): return "Shot image not found", 404
    return send_file(path, mimetype='image/jpeg')

from src.app_helper import get_image, detectionAPI

@app.route('/api/process_image', methods=['POST'])
def process_image():
    f = request.files['image']
    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(filepath)
    
    response = []
    get_image(filepath, filename, response)
    
    return jsonify({'response': convert_numpy(response)})

@app.route('/api/detection_json', methods=['POST'])
def detection_json():
    f = request.files['image']
    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(filepath)
    
    response = []
    detectionAPI(response, filepath)
    
    return jsonify({'response': convert_numpy(response)})

@app.route('/api/detections/<filename>')
def get_detection_image(filename):
    return send_file(f'./static/detections/{filename}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

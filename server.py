import os, time, numpy as np
from flask import Flask, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename
from src.app_helper import getVideoStream
from src.config import shooting_result

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
os.makedirs('./static/uploads', exist_ok=True)
os.makedirs('./static/detections', exist_ok=True)

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
    stream = getVideoStream(request.args.get('video_path'))
    return Response(stream, mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/shooting_result')
def get_shooting_result():
    return jsonify(convert_numpy(shooting_result))

@app.route("/api/download_processed_video")
def download_processed_video():
    if not os.path.exists('./static/detections/final_output.mp4'): return "Video not finished processing", 404
    return send_file('./static/detections/final_output.mp4', mimetype='video/mp4')

from src.app_helper import get_image, detectionAPI

@app.route('/api/process_image', methods=['POST'])
def process_image():
    f = request.files['image']
    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(filepath)
    
    response = {}
    get_image(filepath, filename, response)
    
    return jsonify({'response': convert_numpy(response)})

@app.route('/api/detection_json', methods=['POST'])
def detection_json():
    f = request.files['image']
    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(filepath)
    
    response = {}
    detectionAPI(response, filepath)
    
    return jsonify({'response': convert_numpy(response)})

@app.route('/api/detections/<filename>')
def get_detection_image(filename):
    return send_file(f'./static/detections/{filename}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

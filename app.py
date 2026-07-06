import os
import requests
from flask import Flask, render_template, Response, request, session, jsonify, abort
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.secret_key = "super secret key"

# =========================================================================
# PASTE YOUR NGROK URL HERE (No trailing slash)
# e.g., KAGGLE_GPU_URL = "http://abcdef.ngrok.io"
# =========================================================================
KAGGLE_GPU_URL = "https://gdmmu-34-34-121-231.free.pinggy.net"

# Required to bypass Ngrok's free tier browser warning screen for API requests
NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/detection_json', methods=['GET', 'POST'])
def detection_json():
    if request.method == 'POST':
        f = request.files['image']
        files = {'image': (f.filename, f.read(), f.mimetype)}
        try:
            r = requests.post(f"{KAGGLE_GPU_URL}/api/detection_json", files=files, headers=NGROK_HEADERS)
            return jsonify(r.json()), r.status_code
        except Exception as e:
            print("Error connecting to Kaggle GPU:", e)
            abort(500)

@app.route('/sample_detection', methods=['GET', 'POST'])
def upload_sample_image():
    if request.method == 'POST':
        filename = "sample_image.jpg"
        filepath = "./static/uploads/sample_image.jpg"
        with open(filepath, 'rb') as f:
            files = {'image': (filename, f.read(), 'image/jpeg')}
            r = requests.post(f"{KAGGLE_GPU_URL}/api/process_image", files=files, headers=NGROK_HEADERS)
        
        data = r.json()
        response = data['response']
        # Point the frontend to the image generated on Kaggle
        display_url = f"{KAGGLE_GPU_URL}/api/detections/{filename}"
        return render_template("shot_detection.html", display_detection=display_url, fname=filename, response=response)

@app.route('/basketball_detection', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        f = request.files['image']
        filename = secure_filename(f.filename)
        files = {'image': (filename, f.read(), f.mimetype)}
        r = requests.post(f"{KAGGLE_GPU_URL}/api/process_image", files=files, headers=NGROK_HEADERS)
        
        data = r.json()
        response = data['response']
        display_url = f"{KAGGLE_GPU_URL}/api/detections/{filename}"
        return render_template("shot_detection.html", display_detection=display_url, fname=filename, response=response)

@app.route('/sample_analysis', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        filename = "sample_video.mp4"
        filepath = "./static/uploads/sample_video.mp4"
        with open(filepath, 'rb') as f:
            files = {'video': (filename, f.read(), 'video/mp4')}
            r = requests.post(f"{KAGGLE_GPU_URL}/api/upload_video", files=files, headers=NGROK_HEADERS)
        session['video_path'] = r.json()['video_path']
        return render_template("shooting_analysis.html")

@app.route('/shooting_analysis', methods=['GET', 'POST'])
def upload_sample_video():
    if request.method == 'POST':
        f = request.files['video']
        filename = secure_filename(f.filename)
        files = {'video': (filename, f.read(), f.mimetype)}
        r = requests.post(f"{KAGGLE_GPU_URL}/api/upload_video", files=files, headers=NGROK_HEADERS)
        session['video_path'] = r.json()['video_path']
        return render_template("shooting_analysis.html")

@app.route('/video_feed')
def video_feed():
    video_path = session.get('video_path', None)
    # Stream the multipart frames directly from Kaggle GPU
    req = requests.get(f"{KAGGLE_GPU_URL}/api/video_feed", params={'video_path': video_path}, stream=True, headers=NGROK_HEADERS)
    return Response(req.iter_content(chunk_size=1024), content_type=req.headers.get('Content-Type', 'text/plain'))

@app.route("/result", methods=['GET', 'POST'])
def result():
    r = requests.get(f"{KAGGLE_GPU_URL}/api/shooting_result", headers=NGROK_HEADERS)
    return render_template("result.html", shooting_result=r.json(), KAGGLE_GPU_URL=KAGGLE_GPU_URL)

#disable caching
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=True)


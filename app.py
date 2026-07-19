import os
import requests
from flask import Flask, render_template, Response, request, session, jsonify, abort, render_template_string
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
KAGGLE_GPU_URL = "https://appendix-providence-rob-mails.trycloudflare.com"

# Required to bypass Ngrok/Localtunnel's free tier browser warning screen for API requests
NGROK_HEADERS = {"ngrok-skip-browser-warning": "true", "Bypass-Tunnel-Reminder": "true"}

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
        # Point the frontend to the image generated on Kaggle, proxied locally to bypass pinggy warning
        display_url = f"/api/detections/{filename}"
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
        display_url = f"/api/detections/{filename}"
        return render_template("shot_detection.html", display_detection=display_url, fname=filename, response=response)

@app.route('/sample_analysis', methods=['GET', 'POST'])
def upload_sample_video():
    if request.method == 'POST':
        # Bypass upload since Kaggle already has this sample file cloned
        session['video_path'] = "./static/uploads/sample_video.mp4"
        return render_template("shooting_analysis.html")

@app.route('/shooting_analysis', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        f = request.files['video']
        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)
        
        # Stream the file to bypass Pinggy's strict Content-Length limit
        with open(filepath, 'rb') as video_file:
            files = {'video': (filename, video_file, f.mimetype)}
            r = requests.post(f"{KAGGLE_GPU_URL}/api/upload_video", files=files, headers=NGROK_HEADERS)
            
        session['video_path'] = r.json()['video_path']
        return render_template("shooting_analysis.html")

@app.route('/video_feed')
def video_feed():
    video_path = session.get('video_path', None)
    # Stream the multipart frames directly from Kaggle GPU
    req = requests.get(f"{KAGGLE_GPU_URL}/api/video_feed", params={'video_path': video_path}, stream=True, headers=NGROK_HEADERS)
    return Response(req.iter_content(chunk_size=1024), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detections/<filename>')
def proxy_detection_image(filename):
    req = requests.get(f"{KAGGLE_GPU_URL}/api/detections/{filename}", stream=True, headers=NGROK_HEADERS)
    return Response(req.iter_content(chunk_size=1024), content_type=req.headers.get('Content-Type', 'image/jpeg'))

@app.route('/api/download_processed_video')
def proxy_download_processed_video():
    req = requests.get(f"{KAGGLE_GPU_URL}/api/download_processed_video", stream=True, headers=NGROK_HEADERS)
    return Response(req.iter_content(chunk_size=1024), content_type=req.headers.get('Content-Type', 'video/mp4'))

@app.route("/result", methods=['GET', 'POST'])
def result():
    try:
        # Give Kaggle 5 seconds to respond. If it's still processing the video, the Global Interpreter Lock 
        # (GIL) will block the request and this will intentionally timeout.
        r = requests.get(f"{KAGGLE_GPU_URL}/api/shooting_result", headers=NGROK_HEADERS, timeout=5)
        r.raise_for_status()
        result_data = r.json()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.RequestException):
        # Kaggle is busy! Render a friendly loading page that auto-refreshes.
        return render_template_string("""
            <html>
                <head>
                    <meta http-equiv="refresh" content="5">
                    <style>
                        body { font-family: sans-serif; text-align: center; padding-top: 100px; background-color: #F3ECE1; }
                        .loader { border: 8px solid #f3f3f3; border-top: 8px solid #E65A5A; border-radius: 50%; width: 60px; height: 60px; animation: spin 2s linear infinite; margin: 0 auto; }
                        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                    </style>
                </head>
                <body>
                    <div class="loader"></div>
                    <h2 style="color: #4A4A4A; margin-top: 30px;">AI is still analyzing your video...</h2>
                    <p style="color: #7A7A7A;">Please leave this tab open. This page will automatically refresh every 5 seconds until it's done.</p>
                </body>
            </html>
        """)

    return render_template("result.html", shooting_result=result_data, KAGGLE_GPU_URL=KAGGLE_GPU_URL)

#disable caching
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=True)


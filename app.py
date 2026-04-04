from flask import Flask, render_template, Response, jsonify
from ultralytics.solutions import Heatmap
import cv2
import threading
import time
import random  # demo ke liye extra stats

app = Flask(__name__)

# ====================== GLOBAL VARIABLES ======================
heatmap_obj = None
frame_lock = threading.Lock()
latest_frame = None
current_count = 0
total_today = 1247          # ← Yeh global hai
peak_count = 42
avg_dwell = 87
# ============================================================

def init_heatmap():
    global heatmap_obj
    print("🚀 Loading YOLO12 model + Heatmap solution... (pehle baar download hoga)")
    heatmap_obj = Heatmap(
        model="yolo12n.pt",
        classes=[0],           # sirf person class
        show=False,
        colormap=cv2.COLORMAP_JET,
        conf=0.3,
        iou=0.7,
        tracker="botsort.yaml"
    )
    print("✅ YOLO12 + Heatmap ready!")

def process_video():
    global latest_frame, current_count, total_today   # ← YE LINE ADD KI GAYI HAI (error fix)
    
    video_path = "demo.mp4"
    
    while True:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"🎥 Playing {video_path} ... (looping)")
        
        while True:
            success, im0 = cap.read()
            if not success:
                break
                
            # YOLO12 + Heatmap processing
            results = heatmap_obj(im0)
            annotated = results.plot_im
            
            # Current real-time count
            current_count = getattr(results, 'total_tracks', 0)
            
            # Demo: total_today ko randomly badhao (real project mein database se aayega)
            if random.random() < 0.2:
                total_today += random.randint(1, 3)
            
            # Latest frame ko thread-safe way mein update karo
            with frame_lock:
                latest_frame = annotated.copy()
            
            time.sleep(0.03)   # ~30 FPS feel
        
        cap.release()
        time.sleep(0.5)   # short delay before restarting video

def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify({
        "customers": current_count,
        "total_today": total_today,
        "peak_count": peak_count,
        "avg_dwell": avg_dwell
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    init_heatmap()
    
    # Start video processing in background thread
    thread = threading.Thread(target=process_video, daemon=True)
    thread.start()
    
    print("🌐 Premium Retail Dashboard running → http://127.0.0.1:5000")
    print("📊 Stats: /stats")
    print("📹 Live Feed: /video_feed")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
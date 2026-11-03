from flask import Flask, render_template, request, jsonify, Response, send_file, redirect, url_for
import cv2
import numpy as np
import os
import time
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename

# Import custom utilities
from utils.preprocessing import resize_frame
from utils.vehicle_detection import VehicleTracker, detect_vehicles_in_frame
from utils.congestion_prediction import CongestionPredictor
from utils.signal_controller import SignalController
from utils.analytics import init_db, get_historical_data, generate_charts, write_csv_report, write_pdf_report

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB Max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('processed', exist_ok=True)

# Initialize components
init_db()
predictor = CongestionPredictor()
controller = SignalController()
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
tracker = VehicleTracker(roi_y=240)

# Global runtime state variables
STATE = {
    "video_path": "dataset/traffic_video.mp4",
    "is_live_camera": False,
    "camera_id": 0,
    "rain_mode": False,
    "night_mode": False,
    "emergency_override": False,
    "fps": 0,
    "current_density": 0.0,
    "congestion_level": "No Traffic",
    "congestion_confidence": 1.0,
    "green_time": 20.0,
    "yellow_time": 5.0,
    "red_time": 20.0,
    "signal_mode": "ADAPTIVE",
    "average_wait": 20.0,
    "accident_detected": False,
    "emergency_detected": False,
    "total_vehicles": 0,
    "car_count": 0,
    "bike_count": 0,
    "bus_count": 0,
    "truck_count": 0,
    "active_signal": "RED",  # GREEN, YELLOW, RED
    "signal_timer": 20.0,
    "last_db_log_time": 0.0
}

# Ensure baseline video exists
if not os.path.exists(STATE["video_path"]):
    # If the video isn't compiled yet, we trigger compilation
    try:
        from utils.generate_video import create_synthetic_traffic_video
        create_synthetic_traffic_video(STATE["video_path"])
    except Exception as e:
        print(f"Error creating fallback video: {e}")

def get_db_connection():
    conn = sqlite3.connect("database/traffic.db")
    conn.row_factory = sqlite3.Row
    return conn

def log_to_database():
    """
    Logs current traffic metrics to SQLite database.
    Called periodically during frame processing.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO traffic_data 
            (vehicle_count, car_count, bike_count, bus_count, truck_count, 
             density, congestion_level, green_time, yellow_time, red_time, 
             average_wait, camera_id) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            STATE["total_vehicles"],
            STATE["car_count"],
            STATE["bike_count"],
            STATE["bus_count"],
            STATE["truck_count"],
            STATE["current_density"],
            STATE["congestion_level"],
            STATE["green_time"],
            STATE["yellow_time"],
            STATE["red_time"],
            STATE["average_wait"],
            "CAM-001"
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database logging error: {e}")

def gen_frames():
    """
    MJPEG Stream Generator.
    Reads frames, detects, tracks, predicts, updates signal controller,
    annotates details onto image and yields it.
    """
    global tracker, bg_subtractor
    
    cap = None
    frame_counter = 0
    last_time = time.time()
    
    # Internal signal state tracking inside generator
    local_signal = "GREEN"
    local_timer = 20.0
    
    while True:
        # 1. Capture Source Initialization / Maintenance
        if cap is None:
            if STATE["is_live_camera"]:
                cap = cv2.VideoCapture(int(STATE["camera_id"]))
            else:
                cap = cv2.VideoCapture(STATE["video_path"])
                
            if not cap.isOpened():
                print(f"Error: Could not open video source {STATE['video_path']}")
                time.sleep(2)
                cap = None
                continue

        ret, frame = cap.read()
        
        # Loop video file if ended
        if not ret:
            if not STATE["is_live_camera"]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("Camera feed lost, reconnecting...")
                cap.release()
                cap = None
                time.sleep(2)
                continue
                
        frame_counter += 1
        
        # 2. Resize and Preprocess Frame
        frame = resize_frame(frame, width=640, height=480)
        
        # 3. Vehicle Detection and Centroid Tracking
        detections = detect_vehicles_in_frame(frame, bg_subtractor)
        counts = tracker.update(detections, frame)
        
        # Sync counts to global state
        STATE["car_count"] = counts["car"]
        STATE["bike_count"] = counts["bike"]
        STATE["bus_count"] = counts["bus"]
        STATE["truck_count"] = counts["truck"]
        
        # 4. Density Estimation
        # Road Area representation in pixels: ROI road segment (width ~240px, height ~400px) = 96,000 sq pixels.
        # Estimate density as: (Sum of areas of all active tracked bounding boxes) / (Road Area segment)
        active_boxes_area = 0
        active_count = 0
        for obj_id, obj_data in tracker.tracked_objects.items():
            if obj_data["frames_lost"] == 0:
                bbox = obj_data["bbox"]
                active_boxes_area += bbox[2] * bbox[3]
                active_count += 1
                
        road_area_pixels = 80000.0  # Normalized area of road lanes
        density_pct = min(100.0, (active_boxes_area / road_area_pixels) * 100.0)
        STATE["current_density"] = round(density_pct, 1)
        
        # Estimate speed based on centroid movements (mocking average speed of active objects)
        speeds = []
        for obj_id, obj_data in tracker.tracked_objects.items():
            if obj_data["frames_lost"] == 0 and len(obj_data["speed_history"]) > 0:
                # Convert pixel delta to simulated km/h: speed_px * 10
                speeds.append(obj_data["speed_history"][-1] * 10.0)
                
        avg_speed_kmh = np.mean(speeds) if speeds else 65.0
        # If an accident occurs, speed drops
        if tracker.accident_detected:
            avg_speed_kmh = max(5.0, avg_speed_kmh * 0.15)
            
        # 5. Machine Learning Congestion Prediction
        congestion, conf = predictor.predict(active_count, avg_speed_kmh, density_pct / 100.0)
        STATE["congestion_level"] = congestion
        STATE["congestion_confidence"] = conf
        
        # 6. Adaptive Signal Control Timing
        is_emergency = tracker.emergency_vehicle_active or STATE["emergency_override"]
        STATE["emergency_detected"] = tracker.emergency_vehicle_active
        STATE["accident_detected"] = tracker.accident_detected
        
        timings = controller.calculate_timings(
            density_pct, 
            emergency_active=is_emergency,
            rain_mode=STATE["rain_mode"], 
            night_mode=STATE["night_mode"]
        )
        
        # Sync controller timings
        STATE["green_time"] = timings["green_time"]
        STATE["yellow_time"] = timings["yellow_time"]
        STATE["red_time"] = timings["red_time"]
        STATE["signal_mode"] = timings["mode"]
        
        # Calculate dynamic waiting time based on active count and congestion multiplier
        # waiting_time = red_time + (vehicle_count * 1.5)
        STATE["average_wait"] = round(STATE["red_time"] + (active_count * 1.2), 1)
        
        # Calculate current active counts
        STATE["total_vehicles"] = counts["car"] + counts["bike"] + counts["bus"] + counts["truck"] + counts["ambulance"]
        
        # 7. Internal Traffic Light Signal Timer
        now_time = time.time()
        elapsed = now_time - last_time
        last_time = now_time
        
        # Simple simulated timer ticking down
        local_timer -= elapsed
        if local_timer <= 0:
            # Transition signal
            if local_signal == "GREEN":
                local_signal = "YELLOW"
                local_timer = STATE["yellow_time"]
            elif local_signal == "YELLOW":
                local_signal = "RED"
                local_timer = STATE["red_time"]
            else:
                local_signal = "GREEN"
                local_timer = STATE["green_time"]
                
        STATE["active_signal"] = local_signal
        STATE["signal_timer"] = round(max(0.0, local_timer), 1)
        
        # FPS calculation
        STATE["fps"] = int(1.0 / max(0.001, elapsed))
        
        # 8. Log into Database every 30 frames (~1 second interval)
        if frame_counter % 30 == 0:
            log_to_database()
            
        # 9. Frame Annotations
        # Draw tracked bounding boxes
        for obj_id, obj_data in tracker.tracked_objects.items():
            if obj_data["frames_lost"] == 0:
                x, y, w, h = obj_data["bbox"]
                vtype = obj_data["type"]
                
                # Colors (BGR)
                color = (255, 120, 0) # default blue-ish/orange
                if vtype == "bike": color = (0, 255, 0)
                elif vtype == "car": color = (235, 150, 50)
                elif vtype == "bus": color = (0, 140, 255)
                elif vtype == "truck": color = (120, 120, 120)
                elif vtype == "ambulance": color = (0, 0, 255) # Red for emergency
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Label text
                label = f"{vtype.upper()} #{obj_id}"
                cv2.putText(frame, label, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
        # Draw ROI crossing line
        roi_color = (0, 0, 255) if not is_emergency else (0, 255, 0)
        cv2.line(frame, (80, tracker.roi_y), (560, tracker.roi_y), roi_color, 2)
        cv2.putText(frame, f"ROI LINE ({'EMERGENCY OVERRIDE' if is_emergency else 'ACTIVE'})", 
                    (90, tracker.roi_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, roi_color, 1)
        
        # Draw Accident Indicator if triggered
        if tracker.accident_detected and tracker.accident_location:
            ax, ay = tracker.accident_location
            cv2.circle(frame, (int(ax), int(ay)), 30, (0, 0, 255), 2)
            cv2.putText(frame, "ACCIDENT ALERT", (int(ax) - 50, int(ay) - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        # Draw overlay HUD
        hud_bg = frame.copy()
        cv2.rectangle(hud_bg, (10, 10), (280, 110), (0, 0, 0), -1)
        cv2.addWeighted(hud_bg, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, f"CONGESTION: {STATE['congestion_level']}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"DENSITY: {STATE['current_density']}%", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"SIGNAL: {STATE['active_signal']} ({STATE['signal_timer']}s)", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    (0, 255, 0) if STATE['active_signal']=='GREEN' else ((0, 255, 255) if STATE['active_signal']=='YELLOW' else (0, 0, 255)), 2)
        cv2.putText(frame, f"MODE: {STATE['signal_mode']}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    
        # Encode as JPG and yield
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
               
        # Limit processing frame rate to match input video frame rate ~30 FPS
        time.sleep(0.02)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """
    Saves an uploaded traffic video and overrides the streaming path.
    """
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Update system state
        STATE["video_path"] = filepath
        STATE["is_live_camera"] = False
        
        # Reset tracker & background subtractor
        global tracker, bg_subtractor
        tracker = VehicleTracker(roi_y=240)
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
        
        return redirect(url_for('index'))

@app.route('/live')
def live_stream():
    """
    Returns the live MJPEG processing frame stream.
    """
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process', methods=['GET', 'POST'])
def process_data():
    """
    Exposes and sets system settings and counts.
    """
    if request.method == 'POST':
        data = request.get_json() or {}
        if "rain_mode" in data:
            STATE["rain_mode"] = bool(data["rain_mode"])
        if "night_mode" in data:
            STATE["night_mode"] = bool(data["night_mode"])
        if "emergency_override" in data:
            STATE["emergency_override"] = bool(data["emergency_override"])
        if "is_live_camera" in data:
            STATE["is_live_camera"] = bool(data["is_live_camera"])
            
            # Reset sources
            global tracker, bg_subtractor
            tracker = VehicleTracker(roi_y=240)
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
            
        return jsonify({"status": "updated", "state": STATE})
        
    # GET returns current operational metrics
    return jsonify({
        "vehicle_count": STATE["total_vehicles"],
        "car_count": STATE["car_count"],
        "bike_count": STATE["bike_count"],
        "bus_count": STATE["bus_count"],
        "truck_count": STATE["truck_count"],
        "density": STATE["current_density"],
        "congestion_level": STATE["congestion_level"],
        "congestion_confidence": round(STATE["congestion_confidence"] * 100, 1),
        "green_time": STATE["green_time"],
        "yellow_time": STATE["yellow_time"],
        "red_time": STATE["red_time"],
        "average_wait": STATE["average_wait"],
        "active_signal": STATE["active_signal"],
        "signal_timer": STATE["signal_timer"],
        "signal_mode": STATE["signal_mode"],
        "fps": STATE["fps"],
        "rain_mode": STATE["rain_mode"],
        "night_mode": STATE["night_mode"],
        "emergency_override": STATE["emergency_override"],
        "accident_detected": STATE["accident_detected"],
        "emergency_detected": STATE["emergency_detected"]
    })

@app.route('/reset_counts', methods=['POST'])
def reset_counts():
    global tracker
    tracker = VehicleTracker(roi_y=240)
    return jsonify({"status": "reset"})

@app.route('/analytics')
def analytics():
    charts, is_preview = generate_charts()
    metrics_log = predictor.get_metrics_summary()
    return render_template('analytics.html', charts=charts, is_preview=is_preview, metrics_log=metrics_log)

@app.route('/history')
def history():
    df = get_historical_data(100)
    logs = df.to_dict(orient='records')
    return render_template('history.html', logs=logs)

@app.route('/report')
def report_page():
    return redirect(url_for('analytics'))

@app.route('/download')
def download_report():
    report_type = request.args.get('type', 'pdf').lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if report_type == 'csv':
        filepath = f"processed/traffic_report_{timestamp}.csv"
        write_csv_report(filepath)
        return send_file(filepath, as_attachment=True, download_name=f"traffic_report_{timestamp}.csv")
    else: # PDF
        filepath = f"processed/traffic_report_{timestamp}.pdf"
        write_pdf_report(filepath)
        return send_file(filepath, as_attachment=True, download_name=f"traffic_report_{timestamp}.pdf")

if __name__ == '__main__':
    # use_reloader=False prevents the watchdog from restarting the server
    # whenever files change. This is critical because TensorFlow model loading
    # takes ~10 seconds, and the reloader would restart during that window,
    # creating an infinite restart loop where routes never become available.
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

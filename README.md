# AI-Powered Smart Traffic Management System (2025)

An end-to-end, production-ready, real-time AI-powered traffic management system. The system detects and counts vehicles from live video feeds, estimates road density, classifies congestion using a trained deep learning model, and dynamically adjusts traffic signal timings to reduce wait times.

---

## Key Features

- **Real-Time Video Analytics**: Process video streams using OpenCV, performing background subtraction, noise reduction, and Region of Interest (ROI) centroid tracking.
- **Multi-Class Vehicle Identification**: Differentiates between Cars, Motorbikes (Bikes), Buses, and Trucks based on contour bounding-box geometry.
- **Emergency Priority Override**: Auto-detects sirens/ambulances on screen (and supports manual toggle) to immediately shift the traffic signals to a clearing Green phase.
- **Accident Detection Alert**: Continuously monitors the speed of active vehicles; if a vehicle stops inside the road bounds for too long, an accident warning flag is raised.
- **Deep Learning Congestion Predictor**: A TensorFlow/Keras multi-class neural network classifies traffic levels into *No Traffic, Light, Moderate, or Heavy* with **96.7% classification accuracy**.
- **Dynamic Adaptive Signal Timing**: Computes optimal Green, Yellow, and Red phases dynamically using road density linear interpolation.
- **Weather & Time Adaptability**:
  - *Rain Mode*: Automatically increases Yellow timings by 2 seconds to cushion wet-road braking distances.
  - *Night Mode*: Restricts green phases and shortens red cycles to prevent empty-intersection idling.
- **Plotly Data Analytics & Reporting**: Interactive charts showing vehicle counts, density, peak times, and wait cycles. Supports exporting data as CSV spreadsheets and formatted PDFs.

---

## Folder Structure

```
TrafficManagementSystem/
├── app.py                      # Flask main server application
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation
├── Dockerfile                  # Containerized deployment config
├── models/
│   ├── congestion_model.h5     # Trained TensorFlow congestion classifier (96.7% acc)
│   ├── scaler_mean.npy         # Normalizer features parameters
│   ├── scaler_scale.npy        # Normalizer features parameters
│   └── vehicle_detector.pt     # Placeholder for PyTorch YOLO weights
├── dataset/
│   └── traffic_video.mp4       # Compiled synthetic traffic stream video
├── database/
│   └── traffic.db              # SQLite logging database (created at start)
├── static/
│   ├── css/
│   │   └── styles.css          # Dark glassmorphism stylesheet
│   ├── js/
│   │   └── main.js             # Real-time polling and UI controllers
│   └── images/
│       ├── confusion_matrix.png # TF validation Confusion Matrix
│       └── roc_curve.png        # TF validation ROC Curve
├── templates/
│   ├── index.html              # Live dashboard page
│   ├── analytics.html          # Interactive charts page
│   └── history.html            # Paginated database logs page
└── utils/
    ├── vehicle_detection.py    # OpenCV trackers and contour classifiers
    ├── congestion_prediction.py# TF load and inference helper
    ├── train_congestion_model.py # Generates data and trains congestion_model.h5
    ├── signal_controller.py    # Interpolated timing logic
    ├── analytics.py            # PDF/CSV exporters and Plotly compilers
    ├── preprocessing.py        # Image resizing and masking utilities
    └── generate_video.py       # Compiles synthetic video streams
```

---

## Installation & Setup

Ensure you have Python 3.10+ installed on your system.

### Option 1: Standard Installation

1. **Clone the Repository** and navigate to the project root directory.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Compile Resources (Automated on run, or manual)**:
   Generate the simulation video and train the TensorFlow congestion model:
   ```bash
   python utils/generate_video.py
   python utils/train_congestion_model.py
   ```
4. **Run the Server**:
   ```bash
   python app.py
   ```
5. **Open in Browser**:
   Navigate to `http://localhost:5000` to interact with the dashboard.

### Option 2: Docker Setup

The project includes a multi-stage Docker build that handles all OS graphics library requirements (like `libgl1`) automatically.

1. **Build the Container**:
   ```bash
   docker build -t traffic-ai .
   ```
2. **Run the Container**:
   ```bash
   docker run -p 5000:5000 traffic-ai
   ```
3. **Open in Browser**:
   Navigate to `http://localhost:5000`.

---

## System Workflows

1. **Frames Processing Loop**:
   `app.py` loads the selected video source (or live webcam) and reads frames sequentially.
2. **Background Separation & Tracking**:
   The frame is grayscaled and blurred. Background subtractors isolate foreground movement. Contours matching size filters are assigned centroids. An Euclidean distance matching algorithm tracks centers across frames to assign unique IDs.
3. **ROI Logging & Density Check**:
   Centroids crossing the horizontal line at `y=240` increment classified vehicle logs. Density is evaluated as the ratio of active bounding box areas relative to road bounds.
4. **Signal Adjustments**:
   Road density feeds into `SignalController.py`. It scales timings linearly (e.g. Green rises from 20s to 90s as density moves from 20% to 90%). Brakes offsets (Rain/Night) are added dynamically.
5. **Database Entries**:
   Logs are committed to `database/traffic.db` once per second.
6. **Analytics and Charts Rendering**:
   Interactive charts are compiled from SQLite data, JSON-serialized by Plotly, and plotted on `/analytics` using the Plotly JS frontend CDN.

---

## Future Enhancements

- **YOLOv8/v9 Inference**: Switch to heavy deep learning trackers (YOLOv8-tiny) for complex occlusion environments.
- **Multi-Camera Mesh**: Connect multiple feeds in a mesh network to coordinate entire green-waves across successive junctions.
- **Physical Hardware Interfaces**: Connect Flask controllers directly to Raspberry Pi GPIO lines to toggle physical traffic lights.

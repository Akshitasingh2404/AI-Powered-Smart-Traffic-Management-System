import cv2
import numpy as np
import math

class VehicleTracker:
    def __init__(self, roi_y=240, crossing_deviation=4, max_lost_frames=10):
        self.roi_y = roi_y
        self.crossing_deviation = crossing_deviation
        self.max_lost_frames = max_lost_frames
        
        # Tracked objects: {id: {"centroid": (x, y), "history": [(x,y)], "type": str, "frames_lost": int, "counted": bool, "speed_history": [], "stopped_frames": 0}}
        self.tracked_objects = {}
        self.next_object_id = 1
        
        # Count stats
        self.counts = {
            "car": 0,
            "bike": 0,
            "bus": 0,
            "truck": 0,
            "ambulance": 0
        }
        
        # Flags
        self.accident_detected = False
        self.accident_location = None
        self.emergency_vehicle_active = False

    def get_centroid(self, x, y, w, h):
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        return (cx, cy)

    def classify_vehicle(self, w, h, frame, x, y):
        """
        Classifies the vehicle based on size, aspect ratio, and color cues.
        """
        # Bounding box aspect ratios and sizes in synthetic simulation:
        # bike: 20x30, car: 40x60, ambulance: 42x70, truck: 50x95, bus: 48x110
        if w < 28:
            return "bike"
        elif h > 98:
            return "bus"
        elif h > 78:
            return "truck"
        
        # Differentiating Ambulance from Car
        # An ambulance is mostly white and has a red/blue siren top section.
        # We check the region inside the frame corresponding to this bounding box.
        if 32 <= w <= 48 and 62 <= h <= 76:
            # Crop vehicle area
            crop = frame[max(0, y):min(frame.shape[0], y+h), max(0, x):min(frame.shape[1], x+w)]
            if crop.size > 0:
                # Calculate mean intensity (white detection)
                mean_val = np.mean(crop)
                if mean_val > 180: # Very bright/white body
                    # Look for red or blue colors in the upper half of the crop
                    upper_half = crop[0:int(h/2), :]
                    hsv = cv2.cvtColor(upper_half, cv2.COLOR_BGR2HSV)
                    
                    # Blue siren ranges
                    lower_blue = np.array([100, 100, 100])
                    upper_blue = np.array([140, 255, 255])
                    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
                    
                    # Red siren ranges
                    lower_red1 = np.array([0, 100, 100])
                    upper_red1 = np.array([10, 255, 255])
                    lower_red2 = np.array([170, 100, 100])
                    upper_red2 = np.array([180, 255, 255])
                    red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), 
                                              cv2.inRange(hsv, lower_red2, upper_red2))
                    
                    if np.sum(blue_mask) > 100 or np.sum(red_mask) > 100:
                        return "ambulance"
            return "car"
            
        return "car"

    def update(self, detections, frame):
        """
        Updates tracked objects and registers counts when vehicles cross the ROI line.
        detections: list of bounding boxes [(x, y, w, h), ...]
        """
        current_centroids = []
        for (x, y, w, h) in detections:
            cx, cy = self.get_centroid(x, y, w, h)
            vtype = self.classify_vehicle(w, h, frame, x, y)
            current_centroids.append({"centroid": (cx, cy), "bbox": (x, y, w, h), "type": vtype})
            
        # Match current detections with existing tracked objects
        updated_tracked = {}
        used_detections = set()
        
        # 1. Match based on proximity
        for obj_id, obj_data in self.tracked_objects.items():
            prev_centroid = obj_data["centroid"]
            best_match_idx = -1
            min_dist = float('inf')
            
            for idx, det in enumerate(current_centroids):
                if idx in used_detections:
                    continue
                dist = math.dist(prev_centroid, det["centroid"])
                if dist < 45 and dist < min_dist:  # Distance threshold
                    min_dist = dist
                    best_match_idx = idx
                    
            if best_match_idx != -1:
                det = current_centroids[best_match_idx]
                used_detections.add(best_match_idx)
                
                # Calculate speed (pixels per frame)
                speed = min_dist
                speed_history = obj_data.get("speed_history", [])
                speed_history.append(speed)
                if len(speed_history) > 15:
                    speed_history.pop(0)
                    
                # Check if stopped (accident detection)
                stopped_frames = obj_data.get("stopped_frames", 0)
                # If speed is very low and it is inside active screen area
                if speed < 0.5 and 50 < det["centroid"][1] < 430:
                    stopped_frames += 1
                else:
                    stopped_frames = 0
                    
                # Update tracked object
                updated_tracked[obj_id] = {
                    "centroid": det["centroid"],
                    "bbox": det["bbox"],
                    "history": obj_data["history"] + [det["centroid"]],
                    "type": det["type"],
                    "frames_lost": 0,
                    "counted": obj_data["counted"],
                    "speed_history": speed_history,
                    "stopped_frames": stopped_frames
                }
                
                # Check for crossing the ROI line (y = 240)
                if not obj_data["counted"]:
                    prev_y = prev_centroid[1]
                    curr_y = det["centroid"][1]
                    
                    # Detect crossing from top to bottom or bottom to top
                    crossed = False
                    if prev_y <= self.roi_y <= curr_y:
                        crossed = True
                    elif prev_y >= self.roi_y >= curr_y:
                        crossed = True
                        
                    if crossed:
                        vtype = det["type"]
                        self.counts[vtype] += 1
                        updated_tracked[obj_id]["counted"] = True
            else:
                # If not matched, keep it for a few frames
                if obj_data["frames_lost"] < self.max_lost_frames:
                    obj_data["frames_lost"] += 1
                    updated_tracked[obj_id] = obj_data

        # 2. Register unmatched detections as new objects
        for idx, det in enumerate(current_centroids):
            if idx not in used_detections:
                # Only add if it is not too close to the borders to avoid noise
                cx, cy = det["centroid"]
                if cy < 450 and cy > 30:
                    updated_tracked[self.next_object_id] = {
                        "centroid": det["centroid"],
                        "bbox": det["bbox"],
                        "history": [det["centroid"]],
                        "type": det["type"],
                        "frames_lost": 0,
                        "counted": False,
                        "speed_history": [0],
                        "stopped_frames": 0
                    }
                    self.next_object_id += 1

        self.tracked_objects = updated_tracked
        
        # Post-update audits (Emergency vehicle priority & Accidents)
        self.emergency_vehicle_active = False
        self.accident_detected = False
        self.accident_location = None
        
        for obj_id, obj_data in self.tracked_objects.items():
            # Check for active emergency vehicle
            if obj_data["type"] == "ambulance" and obj_data["frames_lost"] == 0:
                self.emergency_vehicle_active = True
                
            # Check for accidents (vehicle stopped for more than 45 frames)
            if obj_data["stopped_frames"] > 45:
                self.accident_detected = True
                self.accident_location = obj_data["centroid"]
                
        return self.counts

def detect_vehicles_in_frame(frame, bg_subtractor):
    """
    Performs background subtraction and extracts bounding boxes of moving objects.
    """
    if frame is None:
        return []
        
    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(blurred)
    
    # Thresholding & morphological opening/closing to clean noise
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 450: # Filter small noise
            x, y, w, h = cv2.boundingRect(contour)
            
            # Constrain to the lanes (road runs roughly between x=200 and x=440)
            if 180 < (x + w/2) < 460:
                detections.append((x, y, w, h))
                
    return detections

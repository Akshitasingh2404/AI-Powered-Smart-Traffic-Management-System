import cv2
import numpy as np
import os

def create_synthetic_traffic_video(output_path="dataset/traffic_video.mp4", duration_sec=30, fps=30):
    """
    Generates a synthetic MP4 video of traffic to serve as a demo.
    Features:
    - 2 lanes of traffic (one moving down, one moving up)
    - Classified vehicles: Cars (blue), Bikes (green), Buses (orange), Trucks (gray)
    - Flashing emergency ambulance (white with red/blue flashing siren)
    - Accident simulation: a vehicle stops in the middle of the frame
    - ROI Line visible for illustration
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_sec * fps
    
    # Active vehicles in the simulation
    # Directions: 1 = moving down (left lane), -1 = moving up (right lane)
    vehicles = []
    
    # Pre-scheduled vehicle spawns (frame, type, speed, lane/direction, duration_stop, start_stop_frame)
    # Lanes: left lane center = 270, right lane center = 370
    spawns = [
        # (spawn_frame, type, speed, direction, stop_duration, stop_start_frame)
        (15, "car", 4, 1, 0, 0),        # Left lane car
        (45, "bike", 6, -1, 0, 0),      # Right lane bike
        (90, "bus", 3, 1, 0, 0),        # Left lane bus
        (150, "truck", 2.5, -1, 0, 0),  # Right lane truck
        (220, "ambulance", 8, 1, 0, 0), # Emergency vehicle moving fast in left lane
        (300, "car", 4, 1, 150, 400),   # Car that will stop at frame 400 for 150 frames (Accident)
        (350, "bike", 5, 1, 0, 0),      # Left lane bike
        (450, "car", 5, -1, 0, 0),      # Right lane car
        (520, "bus", 3, -1, 0, 0),      # Right lane bus
        (600, "truck", 3, 1, 0, 0),     # Left lane truck
        (680, "car", 4.5, -1, 0, 0),    # Right lane car
        (750, "bike", 6, 1, 0, 0),      # Left lane bike
        (800, "car", 5, -1, 0, 0),      # Right lane car
    ]
    
    # Predefined size parameters
    sizes = {
        "car": (40, 60),       # width, height
        "bike": (20, 30),
        "bus": (48, 110),
        "truck": (50, 95),
        "ambulance": (42, 70)
    }
    
    # Vehicle colors (BGR)
    colors = {
        "car": (230, 124, 115),      # Light Blue
        "bike": (46, 204, 113),     # Green
        "bus": (230, 126, 34),      # Orange
        "truck": (149, 165, 166),   # Gray
        "ambulance": (240, 240, 240) # White
    }

    # Background lane dimensions
    lane_width = 120
    road_left = 320 - lane_width
    road_right = 320 + lane_width
    
    for frame_idx in range(total_frames):
        # Create road background (dark gray)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 40 # Dark gray background
        
        # Draw main road
        cv2.rectangle(frame, (road_left, 0), (road_right, height), (75, 75, 75), -1)
        
        # Draw road shoulders (grass/side)
        cv2.rectangle(frame, (0, 0), (road_left - 10, height), (30, 50, 30), -1)
        cv2.rectangle(frame, (road_right + 10, 0), (width, height), (30, 50, 30), -1)
        
        # Draw shoulder lines (yellow)
        cv2.line(frame, (road_left, 0), (road_left, height), (50, 200, 240), 3)
        cv2.line(frame, (road_right, 0), (road_right, height), (50, 200, 240), 3)
        
        # Draw center dashed lane markings
        dash_len = 20
        gap_len = 20
        for y in range(0, height, dash_len + gap_len):
            offset_y = (y + frame_idx) % (dash_len + gap_len)  # Move lanes slightly for illusion
            cv2.line(frame, (320, y), (320, min(y + dash_len, height)), (240, 240, 240), 2)
            
        # Draw ROI Line (Red, semi-transparent helper in simulation)
        roi_y = 240
        cv2.line(frame, (road_left, roi_y), (road_right, roi_y), (0, 0, 255), 2)
        cv2.putText(frame, "DETECTION ROI LINE", (road_left - 80, roi_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Check spawn schedule
        for spawn in spawns:
            spawn_frame, vtype, speed, direction, stop_duration, stop_start_frame = spawn
            if frame_idx == spawn_frame:
                w, h = sizes[vtype]
                x_center = 260 if direction == 1 else 380
                y_start = -h if direction == 1 else height
                vehicles.append({
                    "id": len(vehicles) + 1,
                    "type": vtype,
                    "speed": speed,
                    "direction": direction,
                    "x": x_center - w//2,
                    "y": y_start,
                    "w": w,
                    "h": h,
                    "stop_duration": stop_duration,
                    "stop_start_frame": stop_start_frame,
                    "stop_counter": 0
                })
                
        # Update and draw vehicles
        active_vehicles = []
        for v in vehicles:
            v_w, v_h = v["w"], v["h"]
            v_type = v["type"]
            v_dir = v["direction"]
            v_speed = v["speed"]
            
            # Check for accident stop
            is_stopped = False
            if v["stop_duration"] > 0 and frame_idx >= v["stop_start_frame"]:
                if v["stop_counter"] < v["stop_duration"]:
                    is_stopped = True
                    v["stop_counter"] += 1
                    
            if not is_stopped:
                v["y"] += v_dir * v_speed
                
            # Keep vehicle if it's still on screen
            if (v_dir == 1 and v["y"] < height) or (v_dir == -1 and v["y"] > -v_h):
                active_vehicles.append(v)
                
                # Draw vehicle box
                top_left = (int(v["x"]), int(v["y"]))
                bottom_right = (int(v["x"] + v_w), int(v["y"] + v_h))
                cv2.rectangle(frame, top_left, bottom_right, colors[v_type], -1)
                cv2.rectangle(frame, top_left, bottom_right, (0,0,0), 2)
                
                # Label vehicle
                label = f"{v_type.upper()}"
                cv2.putText(frame, label, (int(v["x"]), int(v["y"] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Special effects
                if v_type == "ambulance":
                    # Flashing sirens (alternating red/blue circles)
                    siren_color = (0, 0, 255) if (frame_idx // 5) % 2 == 0 else (255, 0, 0)
                    cv2.circle(frame, (int(v["x"] + v_w//4), int(v["y"] + v_h//3)), 4, siren_color, -1)
                    siren_color2 = (255, 0, 0) if (frame_idx // 5) % 2 == 0 else (0, 0, 255)
                    cv2.circle(frame, (int(v["x"] + 3*v_w//4), int(v["y"] + v_h//3)), 4, siren_color2, -1)
                    
                    # Draw a red cross on ambulance
                    cv2.rectangle(frame, (int(v["x"] + v_w//2 - 2), int(v["y"] + v_h//2 - 8)),
                                  (int(v["x"] + v_w//2 + 2), int(v["y"] + v_h//2 + 8)), (0, 0, 255), -1)
                    cv2.rectangle(frame, (int(v["x"] + v_w//2 - 8), int(v["y"] + v_h//2 - 2)),
                                  (int(v["x"] + v_w//2 + 8), int(v["y"] + v_h//2 + 2)), (0, 0, 255), -1)
                
                elif is_stopped:
                    # Draw hazard flashing lights (orange dots)
                    hazard_color = (0, 165, 255) if (frame_idx // 10) % 2 == 0 else (50, 50, 50)
                    cv2.circle(frame, (int(v["x"]), int(v["y"])), 3, hazard_color, -1)
                    cv2.circle(frame, (int(v["x"] + v_w), int(v["y"])), 3, hazard_color, -1)
                    cv2.circle(frame, (int(v["x"]), int(v["y"] + v_h)), 3, hazard_color, -1)
                    cv2.circle(frame, (int(v["x"] + v_w), int(v["y"] + v_h)), 3, hazard_color, -1)
                    cv2.putText(frame, "ACCIDENT!", (int(v["x"] - 15), int(v["y"] - 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                    
        vehicles = active_vehicles
        
        # Add frame details (Time, Camera ID)
        cv2.putText(frame, "CAM-001: LIVE FEED", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        out.write(frame)
        
    out.release()
    print(f"Synthetic video successfully written to {output_path}")

if __name__ == "__main__":
    create_synthetic_traffic_video()

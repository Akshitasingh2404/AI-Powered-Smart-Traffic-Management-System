import cv2
import numpy as np

def resize_frame(frame, width=640, height=None):
    """
    Resizes a frame maintaining aspect ratio if height is not provided, 
    otherwise resizes to explicit width and height.
    """
    if frame is None:
        return None
    
    if height is None:
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        height = int(width / aspect_ratio)
        
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def normalize_frame(frame):
    """
    Normalizes frame pixel intensities to range [0, 1].
    """
    if frame is None:
        return None
    return frame.astype(np.float32) / 255.0

def apply_grayscale_and_blur(frame, kernel_size=(5, 5)):
    """
    Converts a frame to grayscale and applies Gaussian blur.
    Used to prepare frames for background subtraction or contour detection.
    """
    if frame is None:
        return None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, kernel_size, 0)
    return blurred

def get_roi_mask(frame, roi_vertices):
    """
    Applies a polygonal Region of Interest (ROI) mask to a frame.
    roi_vertices is a list of coordinates [(x1, y1), (x2, y2), ...]
    """
    if frame is None or len(roi_vertices) == 0:
        return frame
        
    mask = np.zeros_like(frame)
    points = np.array([roi_vertices], dtype=np.int32)
    
    # Fill mask with white (255) for all color channels
    if len(frame.shape) == 3:
        cv2.fillPoly(mask, points, (255, 255, 255))
    else:
        cv2.fillPoly(mask, points, 255)
        
    # Bitwise AND to mask out everything else
    masked_frame = cv2.bitwise_and(frame, mask)
    return masked_frame

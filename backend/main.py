import math
import time
import os
import cv2
import asyncio
import numpy as np
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from collections import defaultdict, deque
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Configuration
# -------------------------
MAX_POSITION_HISTORY = 10  # Reduced for more stable speed calculation
MIN_SPEED_THRESHOLD = 1    # Lower threshold
MAX_SPEED_THRESHOLD = 120  # Reasonable max for most roads
SPEED_SMOOTHING_WINDOW = 5 # Smooth speed over last N measurements

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "models" / "best.pt")
UPLOAD_DIR = str(BASE_DIR / "uploads")
JOB_DIR = str(BASE_DIR / "jobs")
FRONTEND_DIR = BASE_DIR / "frontend"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(JOB_DIR, exist_ok=True)

# -------------------------
# CALIBRATION HELPER TOOL
# -------------------------
def run_calibration_tool():
    """Interactive calibration tool to get accurate road measurements."""
    
    # Auto-detect video path
    video_path = None
    for path in ["uploads/trim.mp4", "trim.mp4", os.path.join(UPLOAD_DIR, "trim.mp4")]:
        if os.path.exists(path):
            video_path = path
            break
    
    if video_path is None:
        print("\n❌ No video found in uploads folder!")
        print("Please upload a video first or specify the path:")
        video_path = input("Enter video path: ").strip()
        if not os.path.exists(video_path):
            print("❌ Video file not found!")
            return None
    
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, frame_copy
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            print(f"✓ Point {len(points)}: ({x}, {y})")
            
            # Draw point
            cv2.circle(frame_copy, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(frame_copy, f"P{len(points)}", (x+15, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw line between points
            if len(points) > 1:
                cv2.line(frame_copy, tuple(points[-2]), tuple(points[-1]), (0, 255, 255), 2)
            
            cv2.imshow("Calibration", frame_copy)
            
            if len(points) == 4:
                # Close the polygon
                cv2.line(frame_copy, tuple(points[-1]), tuple(points[0]), (0, 255, 255), 2)
                cv2.imshow("Calibration", frame_copy)
                print("\n" + "="*70)
                print("✅ All 4 points selected!")
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ ERROR: Cannot open video at '{video_path}'")
        return None
    
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Cannot read video frame")
        cap.release()
        return None
    
    frame_copy = frame.copy()
    height, width = frame.shape[:2]
    
    print("\n" + "="*70)
    print("🎯 VEHICLE SPEED CALIBRATION TOOL")
    print("="*70)
    print(f"📹 Video: {width}x{height} pixels")
    print("\n📍 INSTRUCTIONS:")
    print("   Click 4 corners of a RECTANGULAR road section in this order:")
    print("   1️⃣  Bottom-left (close to camera)")
    print("   2️⃣  Bottom-right (close to camera)")
    print("   3️⃣  Top-right (far from camera)")  
    print("   4️⃣  Top-left (far from camera)")
    print("\n💡 TIPS:")
    print("   • Choose a flat road section")
    print("   • Select a section where you know the real dimensions")
    print("   • Try to make it as rectangular as possible")
    print("   • Typical road lane width: 3-4 meters")
    print("="*70 + "\n")
    
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", 1280, 720)
    cv2.setMouseCallback("Calibration", mouse_callback)
    cv2.imshow("Calibration", frame_copy)
    
    print("👆 Click on the video window to select points...")
    cv2.waitKey(0)
    
    result = None
    
    if len(points) == 4:
        print("\n" + "="*70)
        print("📏 Enter REAL-WORLD dimensions of the selected area:")
        print("="*70)
        
        try:
            print("\n💡 Typical values:")
            print("   • Single lane width: 3-4 meters")
            print("   • Two lanes width: 7-10 meters")
            print("   • Depth: 15-30 meters (depends on how far you selected)")
            
            road_width = float(input("\n🔹 Width (in meters): "))
            road_depth = float(input("🔹 Depth/Length (in meters): "))
            
            if road_width <= 0 or road_depth <= 0:
                print("❌ Dimensions must be positive!")
                cv2.destroyAllWindows()
                cap.release()
                return None
            
            dst_points = [[0, 0], [road_width, 0], [road_width, road_depth], [0, road_depth]]
            
            print("\n" + "="*70)
            print("✅ CALIBRATION COMPLETE!")
            print("="*70)
            print("\n📋 Calibration values saved!")
            print(f"   Road dimensions: {road_width}m x {road_depth}m")
            print(f"   Source points: {points}")
            print("="*70)
            
            # Visualize result
            result_frame = frame.copy()
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(result_frame, [pts], True, (0, 255, 0), 3)
            
            for i, pt in enumerate(points, 1):
                cv2.circle(result_frame, tuple(pt), 10, (0, 0, 255), -1)
                cv2.putText(result_frame, f"P{i}", (pt[0]+20, pt[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            cv2.putText(result_frame, f"Road: {road_width}m x {road_depth}m", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            print("\n✨ Showing calibration result... Press any key to continue")
            cv2.imshow("Calibration Result", result_frame)
            cv2.waitKey(0)
            
            result = {
                "src_points": np.float32(points),
                "dst_points": np.float32(dst_points),
                "road_width": road_width,
                "road_depth": road_depth
            }
            
        except ValueError:
            print("❌ Invalid input! Please enter numbers only.")
        except KeyboardInterrupt:
            print("\n\n❌ Cancelled by user")
    else:
        print("\n❌ Not enough points selected.")
    
    cv2.destroyAllWindows()
    cap.release()
    return result


# -------------------------
# Perspective Calibration
# -------------------------
# Default calibration (will be overridden if calibration tool is run)
src_points = np.float32([
    [320, 650],   # bottom-left
    [960, 650],   # bottom-right  
    [740, 420],   # top-right
    [540, 420]    # top-left
])

dst_points = np.float32([
    [0, 0],       # bottom-left
    [8, 0],       # bottom-right (8m road width)
    [8, 20],      # top-right (20m depth)
    [0, 20]       # top-left
])

homography_matrix, _ = cv2.findHomography(src_points, dst_points)

# -------------------------
# App + Model
# -------------------------
app = FastAPI(title="Vehicle Speed Detection API")

try:
    model = YOLO(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/jobs", StaticFiles(directory=JOB_DIR), name="jobs")

# -------------------------
# Utils
# -------------------------
def convert_to_h264(in_path, out_path):
    """Convert video to browser-compatible H.264/AAC mp4."""
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")
    
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        out_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Video converted successfully: {out_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        raise RuntimeError(f"Video conversion failed: {e.stderr}")


class VehicleTracker:
    """Manages vehicle tracking and speed calculation with improved accuracy."""
    
    def __init__(self, max_history=MAX_POSITION_HISTORY):
        self.positions = defaultdict(lambda: deque(maxlen=max_history))
        self.speeds = defaultdict(lambda: deque(maxlen=SPEED_SMOOTHING_WINDOW))
        self.stats = defaultdict(lambda: {
            "speeds": [],
            "max_speed": 0,
            "min_speed": float('inf'),
            "frame_count": 0,
            "first_seen": None,
            "last_seen": None
        })
        self.fps = 25
    
    def set_fps(self, fps):
        """Set video FPS for accurate speed calculation."""
        self.fps = fps
    
    def calculate_speed(self, track_id, cx, cy, frame_number):
        """Calculate vehicle speed with improved accuracy."""
        # Transform point to real-world coordinates
        point = np.array([[[float(cx), float(cy)]]], dtype="float32")
        transformed = cv2.perspectiveTransform(point, homography_matrix)[0][0]
        x_curr, y_curr = float(transformed[0]), float(transformed[1])
        
        # Store position
        self.positions[track_id].append((x_curr, y_curr, frame_number))
        
        # Update tracking stats
        if self.stats[track_id]["first_seen"] is None:
            self.stats[track_id]["first_seen"] = frame_number
        self.stats[track_id]["last_seen"] = frame_number
        self.stats[track_id]["frame_count"] += 1
        
        speed = 0
        
        # Need at least 2 positions to calculate speed
        if len(self.positions[track_id]) >= 2:
            # Use positions with some gap for more stable calculation
            gap = min(5, len(self.positions[track_id]) - 1)
            x_prev, y_prev, frame_prev = self.positions[track_id][-gap-1]
            x_curr, y_curr, frame_curr = self.positions[track_id][-1]
            
            # Calculate distance in meters
            dist_m = math.sqrt((x_curr - x_prev) ** 2 + (y_curr - y_prev) ** 2)
            
            # Calculate time difference
            frame_diff = frame_curr - frame_prev
            time_diff = frame_diff / self.fps
            
            if time_diff > 0 and dist_m > 0.1:  # Minimum 10cm movement
                speed = (dist_m / time_diff) * 3.6  # m/s to km/h
                
                # Filter outliers
                if MIN_SPEED_THRESHOLD < speed < MAX_SPEED_THRESHOLD:
                    self.speeds[track_id].append(speed)
                    self.stats[track_id]["speeds"].append(speed)
                    self.stats[track_id]["max_speed"] = max(
                        self.stats[track_id]["max_speed"], speed
                    )
                    self.stats[track_id]["min_speed"] = min(
                        self.stats[track_id]["min_speed"], speed
                    )
        
        # Return smoothed speed (average of last N measurements)
        if len(self.speeds[track_id]) > 0:
            return sum(self.speeds[track_id]) / len(self.speeds[track_id])
        return 0
    
    def get_summary(self):
        """Get aggregated statistics for all vehicles."""
        summary = {}
        for track_id, stats in self.stats.items():
            if stats["speeds"] and len(stats["speeds"]) >= 3:  # Need at least 3 measurements
                speeds = sorted(stats["speeds"])
                summary[int(track_id)] = {
                    "max_speed": round(stats["max_speed"], 2),
                    "min_speed": round(stats["min_speed"], 2),
                    "avg_speed": round(sum(speeds) / len(speeds), 2),
                    "median_speed": round(speeds[len(speeds)//2], 2),
                    "frame_count": stats["frame_count"],
                    "duration_seconds": round((stats["last_seen"] - stats["first_seen"]) / self.fps, 2)
                }
        return summary
    
    def reset(self):
        """Clear all tracking data."""
        self.positions.clear()
        self.speeds.clear()
        self.stats.clear()


# -------------------------
# Routes
# -------------------------
@app.get("/")
def home():
    """Return frontend index.html"""
    frontend_path = FRONTEND_DIR / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {"message": "Vehicle Speed Detection API", "docs": "/docs"}


@app.get("/health")
def health_check():
    """Check if service is healthy."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "ffmpeg_available": check_ffmpeg()
    }


def check_ffmpeg():
    """Check if ffmpeg is installed."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


@app.get("/calibration-info")
def get_calibration_info():
    """Get current calibration settings."""
    return {
        "src_points": src_points.tolist(),
        "dst_points": dst_points.tolist(),
        "road_width_m": float(dst_points[1][0]),
        "road_depth_m": float(dst_points[2][1])
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Process video and detect vehicle speeds."""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"File uploaded: {file.filename} ({len(content)} bytes)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {str(e)}")
    
    # Setup output paths
    processed_filename = f"processed_{os.path.splitext(file.filename)[0]}.mp4"
    temp_path = os.path.join(JOB_DIR, f"temp_{processed_filename}")
    final_path = os.path.join(JOB_DIR, processed_filename)
    
    # Open video
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot open video file")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup video writer
    try:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise Exception("H264 codec not available")
    except:
        logger.warning("H264 codec not available, using mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    # Initialize tracker
    tracker = VehicleTracker()
    tracker.set_fps(fps)
    frame_count = 0
    
    # Draw calibration zone on first frame (for debugging)
    show_calibration_zone = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Draw calibration zone on first few frames
            if show_calibration_zone and frame_count <= 10:
                pts = src_points.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (255, 0, 255), 2)
                cv2.putText(frame, "Calibration Zone", (int(src_points[0][0]), int(src_points[0][1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Run detection and tracking
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            
            if results[0].boxes.id is not None:
                for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    
                    # Use bottom center of bbox for speed calculation (ground point)
                    cx = (x1 + x2) // 2
                    cy = y2  # Bottom of box (closer to actual ground position)
                    
                    speed = tracker.calculate_speed(int(track_id), cx, cy, frame_count)
                    
                    # Color code by speed
                    if speed < 30:
                        color = (0, 255, 0)  # Green - slow
                    elif speed < 50:
                        color = (0, 255, 255)  # Yellow - medium
                    elif speed < 70:
                        color = (0, 165, 255)  # Orange - fast
                    else:
                        color = (0, 0, 255)  # Red - very fast
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw speed label with background
                    label = f"ID:{int(track_id)} {int(speed)} km/h"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), (0, 0, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw tracking point
                    cv2.circle(frame, (cx, cy), 3, (255, 255, 0), -1)
            
            out.write(frame)
            
            if frame_count % 50 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")
    
    finally:
        cap.release()
        out.release()
    
    logger.info(f"Video processing complete: {frame_count} frames")
    
    # Convert to H.264
    if check_ffmpeg():
        try:
            convert_to_h264(temp_path, final_path)
            os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Video conversion failed, using original: {e}")
            os.rename(temp_path, final_path)
    else:
        logger.warning("FFmpeg not available, using mp4v codec")
        os.rename(temp_path, final_path)
    
    # Get summary statistics
    summary = tracker.get_summary()
    
    logger.info(f"Detected {len(summary)} vehicles with valid speeds")
    for vid, stats in summary.items():
        logger.info(f"  Vehicle {vid}: {stats['avg_speed']} km/h avg, {stats['max_speed']} km/h max")
    
    return {
        "message": "Processing complete",
        "video_url": f"http://127.0.0.1:8000/jobs/{processed_filename}",
        "statistics": summary,
        "frames_processed": frame_count,
        "vehicles_detected": len(summary),
        "calibration": {
            "road_width_m": float(dst_points[1][0]),
            "road_depth_m": float(dst_points[2][1])
        }
    }


@app.post("/calibrate")
async def calibrate(points: dict):
    """Update calibration points for perspective transformation.
    
    Example request body:
    {
        "src_points": [[320, 650], [960, 650], [740, 420], [540, 420]],
        "road_width_m": 8,
        "road_depth_m": 20
    }
    """
    try:
        global homography_matrix, src_points, dst_points
        
        src_points = np.float32(points.get("src_points"))
        road_width = points.get("road_width_m", 8)
        road_depth = points.get("road_depth_m", 20)
        
        dst_points = np.float32([
            [0, 0],
            [road_width, 0],
            [road_width, road_depth],
            [0, road_depth]
        ])
        
        homography_matrix, _ = cv2.findHomography(src_points, dst_points)
        
        logger.info(f"Calibration updated: {road_width}m x {road_depth}m")
        
        return {
            "message": "Calibration updated successfully",
            "src_points": src_points.tolist(),
            "dst_points": dst_points.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Calibration failed: {str(e)}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--calibrate":
        print("\n🎯 Starting Calibration Tool...")
        calibration = run_calibration_tool()
        
        if calibration:
            # Update global calibration
            src_points = calibration["src_points"]
            dst_points = calibration["dst_points"]
            homography_matrix, _ = cv2.findHomography(src_points, dst_points)
            
            print("\n✅ Calibration applied! Starting server with new settings...\n")
        else:
            print("\n⚠️  Calibration cancelled. Starting server with default settings...\n")
    
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    print("\n" + "="*70)
    print("🚀 Vehicle Speed Detection API")
    print("="*70)
    print(f"📍 Server: http://127.0.0.1:8000")
    print(f"📖 API Docs: http://127.0.0.1:8000/docs")
    print(f"🎯 To calibrate: python main.py --calibrate")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
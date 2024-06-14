from flask import Flask, render_template, Response
import cv2
import math
from ultralytics import YOLO
import os
import pandas as pd
from datetime import datetime
import time

app = Flask(__name__)

# Configuration
MODEL_PATH = "yolo-Weights/yolov8n.pt"
IP_CAMERA_URL = 'http://192.168.125.216:8080/video'  # Update this URL
DETECTED_IMAGES_DIR = 'detected_images'
EXCEL_FILE = 'capture_log.xlsx'
CAMERA_RETRY_INTERVAL = 5  # Retry interval in seconds

# Load model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Get class names dynamically from the model
classNames = model.names if hasattr(model, 'names') else ["class" + str(i) for i in range(1000)]

def initialize_camera(url):
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Limit buffer size to reduce latency

    start_time = time.time()
    while not cap.isOpened():
        current_time = time.time()
        if current_time - start_time > CAMERA_RETRY_INTERVAL:
            print("Retrying to connect to the camera...")
            cap.release()
            cap = cv2.VideoCapture(url)
            start_time = current_time
        time.sleep(1)

    cap.set(3, 640)
    cap.set(4, 480)
    return cap

# Try to initialize camera
cap = initialize_camera(IP_CAMERA_URL)
if not cap.isOpened():
    print("Error: Unable to connect to IP camera. Trying local webcam...")
    cap = initialize_camera(0)  # Fallback to local webcam

# Create detected_images directory if it doesn't exist
if not os.path.exists(DETECTED_IMAGES_DIR):
    os.makedirs(DETECTED_IMAGES_DIR)

# Initialize the Excel file if it doesn't exist
if not os.path.isfile(EXCEL_FILE):
    df = pd.DataFrame(columns=['Timestamp', 'Image_Name'])
    df.to_excel(EXCEL_FILE, index=False)

def gen_frames():
    global cap
    while True:
        if not cap.isOpened():
            cap = initialize_camera(IP_CAMERA_URL)
            if not cap.isOpened():
                cap = initialize_camera(0)  # Fallback to local webcam
                if not cap.isOpened():
                    print("Error: Unable to connect to any camera.")
                    time.sleep(CAMERA_RETRY_INTERVAL)
                    continue

        success, img = cap.read()
        if not success:
            print("Error: Unable to read from camera, reconnecting...")
            cap.release()
            time.sleep(CAMERA_RETRY_INTERVAL)
            continue

        # Convert BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get model predictions
        results = model(img_rgb)

        # Process results
        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cls = int(box.cls[0])
                class_name = classNames[cls] if cls < len(classNames) else "Unknown"

                if confidence > 0.5:  # Only consider predictions with confidence > 0.5
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(img, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_image', methods=['POST'])
def capture_image():
    global cap
    if not cap.isOpened():
        cap = initialize_camera(IP_CAMERA_URL)
        if not cap.isOpened():
            cap = initialize_camera(0)  # Fallback to local webcam

    success, img = cap.read()
    if success:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        image_name = os.path.join(DETECTED_IMAGES_DIR, f'capture_{timestamp}.jpg')
        cv2.imwrite(image_name, img)

        # Load existing data from Excel file
        if os.path.isfile(EXCEL_FILE):
            df = pd.read_excel(EXCEL_FILE)
        else:
            df = pd.DataFrame(columns=['Timestamp', 'Image_Name'])

        # Append new data
        new_entry = pd.DataFrame([[timestamp, image_name]], columns=['Timestamp', 'Image_Name'])
        df = pd.concat([df, new_entry], ignore_index=True)
        
        # Save back to Excel file
        df.to_excel(EXCEL_FILE, index=False)

    return ('', 204)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

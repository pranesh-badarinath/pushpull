from flask import Flask, render_template_string, Response
import cv2
import RPi.GPIO as GPIO
import time
import torch
import numpy as np

app = Flask(__name__)

# --- 1. Hardware & Model Setup ---

# Motor pins (adjust to match your wiring)
IN1, IN2, IN3, IN4 = 17, 27, 22, 5

# Initialize Camera
# We use a try/except block to handle camera initialization issues gracefully
camera = None
try:
    # Try index 0 first (usually USB cam).
    # If using official Pi Camera ribbon cable with legacy stack, might need different setup.
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Warning: Could not open video source 0. Trying -1.")
        camera = cv2.VideoCapture(-1)
    # Lowering resolution can help improve FPS on Pi when running AI
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
except Exception as e:
    print(f"Error initializing camera: {e}")

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in [IN1, IN2, IN3, IN4]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, False)

# Initialize YOLO Model
print("Loading YOLOv5n model... this might take a minute on a Pi...")
# Using 'ultralytics/yolov5' from torch hub.
# 'pretrained=True' downloads weights if not present.
# Specifying 'cpu' device explicitly is good practice on Pi without specialized hardware.
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, device='cpu')
print("YOLO model loaded successfully!")


# --- 2. Motor Functions ---
def forward():
    GPIO.output(IN1, True); GPIO.output(IN2, False)
    GPIO.output(IN3, True); GPIO.output(IN4, False)

def backward():
    GPIO.output(IN1, False); GPIO.output(IN2, True)
    GPIO.output(IN3, False); GPIO.output(IN4, True)

def left():
    # Tank turn left: Left side back, right side forward
    GPIO.output(IN1, False); GPIO.output(IN2, True)
    GPIO.output(IN3, True); GPIO.output(IN4, False)

def right():
    # Tank turn right: Left side forward, right side back
    GPIO.output(IN1, True); GPIO.output(IN2, False)
    GPIO.output(IN3, False); GPIO.output(IN4, True)

def stop():
    for pin in [IN1, IN2, IN3, IN4]:
        GPIO.output(pin, False)


# --- 3. Video Streaming Generator with YOLO ---
# --- Modified Video Generator with Frame Skipping ---
def gen_frames():
    frame_count = 0
    # How many raw frames to show between detection frames
    # Increase this number for smoother video, decrease for faster detection updates
    SKIP_FRAMES = 4
    last_annotated_frame = None

    while True:
        if camera is None or not camera.isOpened():
            break
        success, frame = camera.read()
        if not success:
            break
        else:
            # Only run AI if frame_count is a multiple of SKIP_FRAMES + 1
            if frame_count % (SKIP_FRAMES + 1) == 0:
                # --- Run heavy AI inference ---
                results = model(frame)
                last_annotated_frame = results.render()[0]
                final_display = last_annotated_frame
            else:
                # --- Skip AI, just show video ---
                # Option A: Show raw video (smoothest)
                final_display = frame

                # Option B: Show the last known detection results superimposed
                # (Looks better but might show lagging boxes behind moving objects)
                # if last_annotated_frame is not None:
                #      final_display = last_annotated_frame
                # else:
                #      final_display = frame

            frame_count += 1

            # Encode whatever we decided to display
            ret, buffer = cv2.imencode('.jpg', final_display)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# --- 4. Flask Routes & HTML ---
html_template = """
<!doctype html>
<html>
<head>
    <title>AI Car Control</title>
    <style>
        body { text-align: center; font-family: sans-serif; background-color: #f0f0f0; }
        h1 { color: #333; }
        #video-container { margin: 20px auto; border: 5px solid #333; display: inline-block;}
        img { width: 100%; max-width: 640px; height: auto; display: block;}
        .controls { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; max-width: 300px; margin: 20px auto; }
        button { padding: 15px; font-size: 18px; background-color: #007BFF; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:active { background-color: #0056b3; }
        .btn-stop { background-color: #dc3545; grid-column: 2; }
        .btn-stop:active { background-color: #c82333; }
        /* Grid positioning */
        .btn-fwd { grid-column: 2; grid-row: 1; }
        .btn-left { grid-column: 1; grid-row: 2; }
        .btn-right { grid-column: 3; grid-row: 2; }
        .btn-back { grid-column: 2; grid-row: 3; }
    </style>
</head>
<body>
    <h1>YOLOv5 Object Detection Car</h1>
    <div id="video-container">
        <img src="{{ url_for('video_feed') }}">
    </div>
    <div class="controls">
        <button class="btn-fwd" onclick="sendCommand('forward')">▲</button>
        <button class="btn-left" onclick="sendCommand('left')">◀</button>
        <button class="btn-stop" onclick="sendCommand('stop')">STOP</button>
        <button class="btn-right" onclick="sendCommand('right')">▶</button>
        <button class="btn-back" onclick="sendCommand('backward')">▼</button>
    </div>

    <script>
        // Function to send commands without reloading page
        function sendCommand(command) {
            fetch('/' + command)
                .then(response => response.text())
                .then(data => console.log(data))
                .catch(error => console.error('Error:', error));
        }

        // Optional: Stop motors when key is lifted (for keyboard control)
        document.addEventListener('keyup', (e) => {
             // Uncomment below if you want keyboard driving
             // if (['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.key)) sendCommand('stop');
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(html_template)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Command Routes (return simple text for AJAX) ---
@app.route("/forward")
def cmd_forward(): forward(); return "Forward OK"
@app.route("/backward")
def cmd_backward(): backward(); return "Backward OK"
@app.route("/left")
def cmd_left(): left(); return "Left OK"
@app.route("/right")
def cmd_right(): right(); return "Right OK"
@app.route("/stop")
def cmd_stop(): stop(); return "Stop OK"


if __name__ == "__main__":
    try:
        # host='0.0.0.0' makes it accessible on your local network
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        # Cleanup hardware when the app closes (Ctrl+C)
        print("Cleaning up GPIO and Camera...")
        stop()
        GPIO.cleanup()
        if camera and camera.isOpened():

            camera.release()

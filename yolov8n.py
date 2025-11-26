from flask import Flask, render_template_string, Response
import cv2
import RPi.GPIO as GPIO
import time
from ultralytics import YOLO

app = Flask(__name__)

# ==========================================
# --- 1. Hardware & Configuration Setup ---
# ==========================================

# --- GPIO Motor Pins ---
# Adjust these to match how your L298N is wired to the Pi
# Current assumption: IN1=17, IN2=27 (Left Motor), IN3=22, IN4=5 (Right Motor)
IN1, IN2, IN3, IN4 = 17, 27, 22, 5

# --- AI & Performance Config ---
# How many raw video frames to show between every AI detection frame.
# Higher number = smoother video but laggier detection boxes.
# 3 is a good balance for Pi 4 CPU (runs AI on 1 out of every 4 frames).
SKIP_FRAMES = 3
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

# --- Initialize GPIO ---
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in [IN1, IN2, IN3, IN4]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, False)

# --- Initialize Camera ---
camera = None
try:
    # Try index 0 (usually USB). Try -1 if using official PiCam ribbon cable older OS.
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Warning: Camera index 0 failed. Trying index -1.")
        camera = cv2.VideoCapture(-1)

    # IMPORTANT: Low resolution is key for Pi CPU performance
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    # Lower buffer size to reduce internal video lag
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception as e:
    print(f"Error initializing camera: {e}")

# --- Initialize YOLOv8 Model ---
print("Loading YOLOv8 Nano model... (this may download weights on first run)")
# Using yolov8n.pt (Nano) for best speed on CPU
model = YOLO('yolov8n.pt')
print("Model loaded!")


# ==========================================
# --- 2. Motor Control Functions ---
# ==========================================
def forward():
    GPIO.output(IN1, True); GPIO.output(IN2, False)
    GPIO.output(IN3, True); GPIO.output(IN4, False)

def backward():
    GPIO.output(IN1, False); GPIO.output(IN2, True)
    GPIO.output(IN3, False); GPIO.output(IN4, True)

def left():
    # Tank turn left
    GPIO.output(IN1, False); GPIO.output(IN2, True)
    GPIO.output(IN3, True); GPIO.output(IN4, False)

def right():
    # Tank turn right
    GPIO.output(IN1, True); GPIO.output(IN2, False)
    GPIO.output(IN3, False); GPIO.output(IN4, True)

def stop():
    for pin in [IN1, IN2, IN3, IN4]:
        GPIO.output(pin, False)


# ==========================================
# --- 3. Video Generator with AI Skipping ---
# ==========================================
def gen_frames():
    frame_count = 0
    last_annotated_frame = None

    while True:
        if camera is None or not camera.isOpened():
            break

        success, frame = camera.read()
        if not success:
            break

        # --- Frame Skipping Logic ---
        # Only run heavy AI inference if the counter hits the target interval
        if frame_count % (SKIP_FRAMES + 1) == 0:
            # Run YOLOv8 inference on the CPU
            # verbose=False stops it from printing detections to terminal continually
            results = model(frame, device='cpu', verbose=False)

            # Plot the results onto the frame
            annotated_frame = results[0].plot()
            last_annotated_frame = annotated_frame
            final_display = annotated_frame
        else:
            # Not an AI frame cycle. By default, just show the raw frame.
            # Alternatively, uncomment below to keep showing the *last* known detection boxes
            # if last_annotated_frame is not None:
            #      final_display = last_annotated_frame
            # else:
            #      final_display = frame
            final_display = frame

        frame_count += 1

        # Encode whatever we decided to display into JPEG for the browser
        ret, buffer = cv2.imencode('.jpg', final_display)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ==========================================
# --- 4. Flask HTML Template & Routes ---
# ==========================================
html_template = """
<!doctype html>
<html>
<head>
    <title>YOLOv8 Robot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { text-align: center; font-family: sans-serif; background-color: #222; color: #eee; touch-action: manipulation; }
        h1 { margin-bottom: 10px; }
        #video-container { margin: 0 auto; border: 3px solid #555; display: inline-block; line-height: 0;}
        img { width: 100%; max-width: 640px; height: auto; }
        .controls { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; max-width: 300px; margin: 20px auto; }
        button { padding: 20px; font-size: 24px; background-color: #444; color: white; border: none; border-radius: 10px; cursor: pointer; -webkit-tap-highlight-color: transparent; }
        button:active { background-color: #666; transform: scale(0.98); }
        .btn-stop { background-color: #d9534f; grid-column: 2; font-weight: bold;}
        .btn-stop:active { background-color: #c9302c; }
        .btn-fwd { grid-column: 2; grid-row: 1; }
        .btn-left { grid-column: 1; grid-row: 2; }
        .btn-right { grid-column: 3; grid-row: 2; }
        .btn-back { grid-column: 2; grid-row: 3; }
    </style>
</head>
<body>
    <h1>YOLOv8 Nano Bot</h1>
    <div id="video-container">
        <img src="{{ url_for('video_feed') }}">
    </div>
    <div class="controls">
        <button class="btn-fwd" ontouchstart="sendCommand('forward')" onmousedown="sendCommand('forward')">▲</button>
        <button class="btn-left" ontouchstart="sendCommand('left')" onmousedown="sendCommand('left')">◀</button>
        <button class="btn-stop" ontouchstart="sendCommand('stop')" onmousedown="sendCommand('stop')">STOP</button>
        <button class="btn-right" ontouchstart="sendCommand('right')" onmousedown="sendCommand('right')">▶</button>
        <button class="btn-back" ontouchstart="sendCommand('backward')" onmousedown="sendCommand('backward')">▼</button>
    </div>

    <script>
        // Using fetch to send commands without reloading the page
        // Added touchstart/mousedown for better mobile responsiveness
        function sendCommand(command) {
            fetch('/cmd/' + command)
                .catch(error => console.error('Error:', error));
        }

        // Optional: Stop on mouse/touch release for "dead man switch" style control
        // Uncomment below to enable stop-on-release
        /*
        document.addEventListener('mouseup', () => sendCommand('stop'));
        document.addEventListener('touchend', () => sendCommand('stop'));
        */
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(html_template)

@app.route("/video_feed")
def video_feed():
    # Return the multipart mixed response for streaming MJPEG
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Unified command route using a path parameter
@app.route("/cmd/<direction>")
def command(direction):
    if direction == "forward": forward()
    elif direction == "backward": backward()
    elif direction == "left": left()
    elif direction == "right": right()
    elif direction == "stop": stop()
    return "", 204 # Return "No Content" success code so browser does nothing

# ==========================================
# --- 5. Main Execution & Cleanup ---
# ==========================================
if __name__ == "__main__":
    try:
        # host='0.0.0.0' makes it accessible on local network IPs
        # debug=False and threaded=True are best for performance
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        # This block runs when you press Ctrl+C to exit
        print("\nShutting down...")
        stop()
        GPIO.cleanup()
        if camera and camera.isOpened():
            camera.release()
        print("Cleanup complete.")

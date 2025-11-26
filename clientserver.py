# ==========================================
# SAVE AS: pi_client.py (RUN ON RASPBERRY PI)
# ==========================================
import cv2
import requests
import RPi.GPIO as GPIO
import time

# --- CONFIGURATION ---
# REPLACE WITH YOUR PC'S IP ADDRESS!
PC_SERVER_URL = "http://192.168.1.X:8000/detect"

# GPIO Pins setup (same as before)
IN1, IN2, IN3, IN4 = 17, 27, 22, 5
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in [IN1, IN2, IN3, IN4]:
    GPIO.setup(pin, GPIO.OUT)

# Motor Functions (simplified for brevity)
def forward(): print(">>> Motors: FORWARD"); # Add real GPIO code here
def left(): print("<<< Motors: LEFT"); # Add real GPIO code here
def right(): print(">>> Motors: RIGHT"); # Add real GPIO code here
def stop(): print("--- Motors: STOP"); # Add real GPIO code here

# Setup Camera
camera = cv2.VideoCapture(0)
# Use low resolution to speed up network transfer!
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print(f"Connecting to AI Server at {PC_SERVER_URL}...")

try:
    while True:
        success, frame = camera.read()
        if not success: break

        # 1. Encode frame to JPEG format for sending
        _, img_encoded = cv2.imencode('.jpg', frame)

        # 2. Define the payload to send to the PC
        files = {'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}

        try:
            # RECORD START TIME for latency check
            start_time = time.time()

            # 3. Send image to PC and wait for response (BLOCKING)
            response = requests.post(PC_SERVER_URL, files=files, timeout=2)

            # Calculate round-trip latency
            latency = round((time.time() - start_time) * 1000) # ms

            # 4. Process response
            if response.status_code == 200:
                data = response.json()
                action = data.get("action", "stop")
                print(f"Latency: {latency}ms | Command received: {action.upper()}")

                # 5. Execute Motor Command
                if action == "forward": forward()
                elif action == "left": left()
                elif action == "right": right()
                else: stop()

        except requests.exceptions.RequestException as e:
            print(f"Network Error: connection to PC lost. Stopping motors.")
            stop()
            time.sleep(1) # Wait a bit before trying again

finally:
    stop()
    GPIO.cleanup()
    camera.release()

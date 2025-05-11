import numpy as np
import cv2
import os
import time
from gtts import gTTS
import threading
import pygame
from ultralytics import YOLO

# Initialize pygame for audio playback
pygame.mixer.init()

# Load YOLOv11 model
print("[INFO] Loading YOLOv11 model...")
model = YOLO("yolo11n.pt")  # Nano model; consider 'yolo11s.pt' for better accuracy

# Get COCO class labels from the model
LABELS = model.names

# Initialize colors for bounding boxes
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables for audio feedback control
last_audio_time = 0
audio_interval = 5  # Seconds between audio playbacks
last_description = ""
last_objects = set()
audio_lock = threading.Lock()  # Lock to prevent overlapping audio

# Function to play audio in a separate thread
def play_audio(description):
    if description and audio_lock.acquire(blocking=False):  # Only play if lock is acquired
        try:
            timestamp = int(time.time() * 1000)
            audio_file = f"ZERO{description[:10]}_{timestamp}.mp3"
            tts = gTTS(text=description, lang="en", slow=False)
            tts.save(audio_file)
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            pygame.mixer.music.set_endevent(pygame.USEREVENT)
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            os.remove(audio_file)
        except PermissionError as e:
            print(f"PermissionError: Could not save audio file: {e}")
        except Exception as e:
            print(f"Error playing audio: {e}")
        finally:
            audio_lock.release()

# Main loop for real-time detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    (H, W) = frame.shape[:2]

    # Perform YOLOv11 inference
    start = time.time()
    results = model(frame, conf=0.25, iou=0.3)  # Confidence threshold = 0.25
    end = time.time()
    print(f"[INFO] YOLOv11 inference took {end - start:.2f} seconds")

    # Initialize lists for detections
    boxes = []
    confidences = []
    class_ids = []

    # Process detections
    for result in results:
        for det in result.boxes:
            x, y, w, h = det.xywh[0].cpu().numpy()
            x = int(x - w / 2)
            y = int(y - h / 2)
            w, h = int(w), int(h)
            confidence = det.conf.cpu().numpy()
            class_id = int(det.cls.cpu().numpy())

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # Generate description and audio
    current_time = time.time()
    descriptions = []
    current_objects = set()

    if len(boxes) > 0:
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            centerX = (2 * x + w) // 2
            centerY = (2 * y + h) // 2

            # Simplified position logic to reduce jitter
            W_pos = "left" if centerX <= W/3 else "center" if centerX <= (W/3 * 2) else "right"
            H_pos = "top" if centerY <= H/3 else "middle" if centerY <= (H/3 * 2) else "bottom"

            object_label = LABELS[class_ids[i]]
            confidence = confidences[i]
            desc = f"{H_pos} {W_pos} {object_label}"
            descriptions.append(desc)
            current_objects.add(object_label)

            # Debugging: Log each detected object
            print(f"Detected: {object_label} at {H_pos} {W_pos} (confidence: {confidence:.2f})")

            # Draw bounding box and label
            color = COLORS[class_ids[i]].tolist()
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label_text = f"{object_label} {confidence:.2f}"
            cv2.putText(frame, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    current_description = ", ".join(sorted(descriptions)) if descriptions else "No objects detected"

    # Log descriptions for debugging
    print(f"Descriptions: {descriptions}")
    print(f"Current description: {current_description}")

    # Trigger audio if interval elapsed or new objects/description
    if (current_time - last_audio_time >= audio_interval) or (current_description != last_description):
        if current_description != last_description or current_objects != last_objects:
            print(f"Audio triggered: {current_description}")
            threading.Thread(target=play_audio, args=(current_description,), daemon=True).start()
            last_audio_time = current_time
            last_description = current_description
            last_objects = current_objects

    # Display the frame
    cv2.imshow("Real-Time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
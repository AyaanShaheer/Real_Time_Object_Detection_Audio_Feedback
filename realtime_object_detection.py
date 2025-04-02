import numpy as np
import cv2
import os
import time
from gtts import gTTS
import threading
import pygame

# Initialize pygame for audio playback
pygame.mixer.init()

# Paths to YOLO files
yolo_dir = "yolo"
weights_path = os.path.join(yolo_dir, "yolov3.weights")
config_path = os.path.join(yolo_dir, "yolov3.cfg")
labels_path = os.path.join(yolo_dir, "coco.names")

# Load COCO class labels
with open(labels_path, "r") as f:
    LABELS = f.read().strip().split("\n")

# Initialize colors for bounding boxes
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Load YOLO model
print("[INFO] Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Get YOLO output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables for audio feedback control
last_audio_time = 0
audio_interval = 5
last_description = ""
last_objects = set()

# Function to play audio in a separate thread
def play_audio(description):
    if description:
        try:
            timestamp = int(time.time() * 1000)
            audio_file = f"output_{timestamp}.mp3"
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

# Main loop for real-time detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    (H, W) = frame.shape[:2]

    # Prepare frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layer_outputs = net.forward(output_layers)
    end = time.time()
    print(f"[INFO] YOLO took {end - start:.2f} seconds")

    # Initialize lists for detections
    boxes = []
    confidences = []
    class_ids = []

    # Process detections
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Generate description and audio
    current_time = time.time()
    descriptions = []
    current_objects = set()

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            centerX = (2 * x + w) // 2
            centerY = (2 * y + h) // 2

            W_pos = "left" if centerX <= W/3 else "center" if centerX <= (W/3 * 2) else "right"
            H_pos = "top" if centerY <= H/3 else "mid" if centerY <= (H/3 * 2) else "bottom"

            object_label = LABELS[class_ids[i]]
            desc = f"{H_pos} {W_pos} {object_label}"
            descriptions.append(desc)
            current_objects.add(object_label)

            color = COLORS[class_ids[i]].tolist()
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, object_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    current_description = ", ".join(descriptions) if descriptions else "No objects detected"

    # Check for significant changes
    significant_change = False
    if current_objects != last_objects:
        significant_change = True
    else:
        if descriptions and last_description:
            current_parts = current_description.split(", ")
            last_parts = last_description.split(", ")
            if len(current_parts) == len(last_parts):
                for curr, last in zip(current_parts, last_parts):
                    curr_pos = curr.split(" ")[0:2]
                    last_pos = last.split(" ")[0:2]
                    if curr_pos[0] != last_pos[0]:
                        significant_change = True
                        break
                    if curr_pos[1] != last_pos[1]:
                        significant_change = True
                        break

    if (current_time - last_audio_time >= audio_interval) or (significant_change and current_description != last_description):
        print(f"Detected: {current_description}")
        threading.Thread(target=play_audio, args=(current_description,), daemon=True).start()
        last_audio_time = current_time
        last_description = current_description
        last_objects = current_objects

    cv2.imshow("Real-Time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
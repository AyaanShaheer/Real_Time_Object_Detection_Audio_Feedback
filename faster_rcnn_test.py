import numpy as np
import cv2
import os
import time
from gtts import gTTS
import threading
import pygame
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Initialize pygame for audio playback
pygame.mixer.init()

# Define COCO labels (80 classes, excluding background)
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Initialize colors for bounding boxes
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_LABELS), 3), dtype="uint8")

# Load Faster R-CNN model
print("[INFO] Loading Faster R-CNN model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

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
            audio_file = f"output_{description[:10]}_{timestamp}.mp3"
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

# Transform for Faster R-CNN input
transform = T.Compose([T.ToTensor()])

# Main loop for real-time detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    (H, W) = frame.shape[:2]

    # Prepare frame for Faster R-CNN
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_tensor = transform(img).to(device)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Perform Faster R-CNN inference
    start = time.time()
    with torch.no_grad():
        predictions = model(img_tensor)
    end = time.time()
    print(f"[INFO] Faster R-CNN inference took {end - start:.2f} seconds")

    # Initialize lists for detections
    boxes = []
    confidences = []
    class_ids = []

    # Process detections
    for box, score, label in zip(predictions[0]["boxes"], predictions[0]["scores"], predictions[0]["labels"]):
        if score > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            w = x2 - x1
            h = y2 - y1
            x = x1
            y = y1
            confidence = score.cpu().numpy()
            class_id = label.cpu().numpy() - 1  # Subtract 1 to align with COCO_LABELS (0-based)

            if class_id < len(COCO_LABELS):  # Ensure valid class ID
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

            # Simplified position logic
            W_pos = "left" if centerX <= W/3 else "center" if centerX <= (W/3 * 2) else "right"
            H_pos = "top" if centerY <= H/3 else "middle" if centerY <= (H/3 * 2) else "bottom"

            object_label = COCO_LABELS[class_ids[i]]
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
# Object Detection with Voice Feedback

This project implements a real-time object detection system with audio feedback to assist visually impaired individuals. It uses the YOLOv11, YOLOv3, Faster R-CNN Model algorithm to detect objects in a live webcam feed and provides audio descriptions of the objects and their positions (e.g., "mid center person").

## Features
- Real-time object detection using YOLOv3, YOLOv11, Faster R-CNN and OpenCV.
- Audio feedback using Google Text-to-Speech (gTTS) and pygame for playback.
- Position-based descriptions (e.g., "top left chair").
- Optimized to reduce repetitive audio feedback.

## Prerequisites
- Python 3.7+
- A webcam
- The following YOLOv3 files:
  - `yolov3.weights` (download from [here](https://pjreddie.com/media/files/yolov3.weights))
  - `yolov3.cfg` (included in the `yolo/` directory)
  - `coco.names` (included in the `yolo/` directory)

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/yolo_project_sem.git
   cd yolo_project_sem

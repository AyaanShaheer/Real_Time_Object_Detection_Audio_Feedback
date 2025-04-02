# Real-Time Object Detection with Audio Feedback 🎙️📷

![Project Banner](https://via.placeholder.com/1200x300.png?text=Real-Time+Object+Detection+with+Audio+Feedback) 

![GitHub repo size](https://img.shields.io/github/repo-size/AyaanShaheer/Real_Time_Object_Detection_Audio_Feedback) ![GitHub last commit](https://img.shields.io/github/last-commit/AyaanShaheer/Real_Time_Object_Detection_Audio_Feedback) ![License](https://img.shields.io/github/license/AyaanShaheer/Real_Time_Object_Detection_Audio_Feedback)

This project implements a **real-time object detection system with audio feedback** to assist visually impaired individuals. Using the YOLOv3 algorithm, it detects objects in a live webcam feed and provides audio descriptions of the objects and their positions (e.g., "mid center person"). The system is designed to enhance independence and mobility for visually impaired users by helping them perceive their surroundings through audio cues.

## 🚀 Features
- **Real-Time Detection**: Detects objects in a live webcam feed using YOLOv3 and OpenCV.
- **Audio Feedback**: Provides voice descriptions of detected objects and their positions using Google Text-to-Speech (gTTS) and pygame.
- **Position Awareness**: Describes object locations (e.g., "top left chair") for better spatial understanding.
- **Optimized Feedback**: Reduces repetitive audio announcements by triggering speech only on significant changes or at intervals.

## 📋 Table of Contents
- [Demo](#demo)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## 🎥 Demo
<!-- Add a demo video or GIF here -->
*Coming soon: A demo video showcasing real-time object detection and audio feedback.*

In the meantime, here’s a screenshot of the system in action:

![Demo Screenshot](https://via.placeholder.com/800x400.png?text=Demo+Screenshot) <!-- Replace with an actual screenshot -->

## 🛠️ Prerequisites
To run this project, you’ll need:
- **Python 3.7+**
- A **webcam** (built-in or external)
- The following YOLOv3 files:
  - `yolov3.weights` (download from [here](https://pjreddie.com/media/files/yolov3.weights))
  - `yolov3.cfg` (included in the `yolo/` directory)
  - `coco.names` (included in the `yolo/` directory)

## ⚙️ Setup Instructions
Follow these steps to set up and run the project on your machine:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AyaanShaheer/Real_Time_Object_Detection_Audio_Feedback.git
   cd Real_Time_Object_Detection_Audio_Feedback

2. **Create a Virtual Environment**:
   ```bash
   python -m venv myenv
   .\myenv\Scripts\activate  # On Windows 
   # source myenv/bin/activate  # On Mac/Linux

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt

4. **Download YOLOv3 Weights**:
   Download yolov3.weights from https://pjreddie.com/media/files/yolov3.weights and place it in the yolo/ directory.

5. **Run the Script**:
    ```bash
     python realtime_object_detection.py

📖 Usage
The script will open your webcam and start detecting objects in real-time.
Audio feedback will describe the detected objects and their positions (e.g., "mid center person, top right chair").
Press q to quit the application.

📂 Project Structure
Real_Time_Object_Detection_Audio_Feedback/
├── yolo/
│   ├── yolov3.cfg      # YOLOv3 configuration file
│   ├── coco.names      # COCO class labels
├── realtime_object_detection.py  # Main script
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── .gitignore          # Git ignore file
Note: The yolov3.weights file is not included in the repository due to its large size (248 MB). Download it as per the setup instructions.


📊 Performance
Frame Rate: ~3 FPS on a CPU with YOLOv3 (can be improved with YOLOv3-Tiny).
Detection Accuracy: Uses the MS-COCO dataset with 80 classes, achieving reasonable accuracy for common objects (e.g., person, chair, car).
Audio Feedback: Optimized to provide updates every 5 seconds or on significant changes, reducing repetition.


🔮 Future Improvements
Switch to YOLOv3-Tiny or YOLOv5 for better real-time performance.
Integrate with a Raspberry Pi for a portable, wearable device.
Add face recognition to identify known individuals.
Include distance estimation using a depth camera (e.g., Intel RealSense).
Train a custom YOLO model with a dataset tailored for visually impaired users.

🙏 Acknowledgments
YOLOv3: Joseph Redmon and the Darknet team (https://pjreddie.com/darknet/yolo/)
Libraries: OpenCV, gTTS, and pygame


📜 License
This project is licensed under the  (MIT LICENSE).

Built with ❤️ by ![Ayaan Shaheer](https://github.com/AyaanShaheer)




# AI-Powered-Real-Time-Object-Detection-and-Distance-Estimation-System-
An AI-powered real-time object detection system using YOLOv8 that identifies multiple objects, maintains consistent tracking with unique IDs, and estimates distances through camera calibration. GPU-optimized for high-performance applications in surveillance, robotics, and autonomous systems requiring precise environmental awareness.


AI-Powered Real-Time Object Detection and Distance Estimation
This project implements an advanced object detection, tracking, and distance estimation system using YOLOv8. It detects multiple objects in real-time, assigns unique IDs, tracks their movement, and estimates distances using camera calibration. The system is optimized for GPU acceleration (CUDA) and supports fallback to CPU.

Features
Real-time object detection using YOLOv8

Multi-object tracking with unique IDs

Distance estimation using known object dimensions

Camera calibration for improved accuracy

High FPS performance with GPU acceleration

Works with live camera or sample video input

Supports an extended set of custom class labels

Tech Stack
Python 3

OpenCV

NumPy

Ultralytics YOLOv8

PyTorch

SciPy

Imutils

Installation
Clone the repository:

text
git clone https://github.com/yourusername/ai-object-detection.git
cd ai-object-detection
Create and activate a virtual environment (recommended):

text
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
Install dependencies:

text
pip install -r requirements.txt
Usage
Run the detector:

text
python detector.py
Press q to quit.

If no camera is found, it automatically loads sample.mp4.

Change the YOLO model in AdvancedObjectDetector(model_path='yolov8n.pt') if needed.

Example Output
Objects detected with bounding boxes, labels, and unique IDs.

Distance (in meters) displayed for known objects (e.g., person, car, laptop).

Real-time FPS displayed on screen.

Future Improvements
Add support for multi-camera input

Integrate with ROS (Robot Operating System) for robotics

Enhance distance estimation with stereo vision

Implement alert system for restricted zones

License
This project is licensed under the MIT License – feel free to use and modify.

Author
Developed by Ravi Der
Passionate about Computer Vision, AI, and Robotics

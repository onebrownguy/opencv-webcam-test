# OpenCV Webcam Face Detection

This project uses **OpenCV** to detect faces in real-time using a webcam.  
It utilizes the **Haar Cascade classifier** for face detection.

## Features
- Captures live webcam feed
- Detects **faces**, **eyes**, **smiles**, and **glasses** using Haar cascades
- Draws bounding boxes:
  - **Green** → Face
  - **Blue** → Eyes
  - **Red** → Smile
- Displays **FPS (frames per second)*
- Detects faces using Haar cascades
- Draws bounding boxes around detected faces
- Press **`q`** to exit

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/opencv-webcam-test.git
   cd opencv-webcam-test

## Files Included
- **`face_detection.py`** → Real-time face, eye, and smile detection script
- **`glasses_detection.py`** → Detects if a person is wearing glasses
- **`haarcascade_frontalface_default.xml`** → Face detection model
- **`haarcascade_eye.xml`** → Eye detection model
- **`haarcascade_smile.xml`** → Smile detection model
- **`haarcascade_eye_tree_eyeglasses.xml`** → Glasses detection model
- **`README.md`** → Documentation

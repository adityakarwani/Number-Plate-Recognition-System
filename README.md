# Number-Plate-Recognition-System

# Plate Recognition System

This project implements a License Plate Recognition (LPR) system using OpenCV, Tesseract OCR, and a pre-trained Haar Cascade classifier for plate detection. The system is designed to detect and recognize license plates from both static images and video streams.

## Features

- **Plate Detection**: Utilizes Haar Cascade Classifier to detect license plates from images and video.
- **OCR Processing**: Extracts and recognizes text from detected plates using Tesseract OCR.
- **Preprocessing**: Enhances the input images and the detected plate regions for better detection and OCR accuracy.
- **Real-time Video Processing**: Can process video streams frame-by-frame and recognize plates in real-time.

## Requirements

To run the project, you need to have the following dependencies installed:

- Python 3.x
- OpenCV (`cv2`)
- Tesseract OCR
- Numpy
- Regex

You can install the necessary Python libraries using pip:

```bash
pip install opencv-python pytesseract numpy

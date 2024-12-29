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

## How It Works

### 1. Image Preprocessing
The input image is first converted to grayscale to simplify the process of detecting the license plate. To reduce noise while preserving edges, a bilateral filter is applied. Following this, adaptive histogram equalization is used to improve the visibility of the plate region, making the plate more distinguishable for detection and OCR.

### 2. Plate Detection
A pre-trained Haar Cascade Classifier (`haarcascade_russian_plate_number.xml`) is used to detect license plates within the image. The classifier is applied to the preprocessed image to locate potential plate regions, identifying where the plate is located in the image.

### 3. OCR Text Extraction
Once a plate is detected, the region containing the plate is passed through additional preprocessing steps to improve text visibility. These steps include resizing the plate region if necessary, applying thresholding for better contrast, and using morphological operations to clean up the image. After these enhancements, Tesseract OCR is used to extract the text from the processed plate image, providing the license plate number.

### 4. Real-Time Video Processing
The system can also process video streams in real-time. For each frame in the video, the system detects and recognizes any license plates, displaying the results live. The process is similar to the image processing, but done continuously for each frame of the video.


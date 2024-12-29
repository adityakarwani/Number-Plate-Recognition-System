import cv2
import numpy as np

class LicensePlateDetector:
    def __init__(self, cascade_path, video_path):
        self.plate_detector = cv2.CascadeClassifier(cascade_path)
        self.video_capture = cv2.VideoCapture(video_path)
        
        if not self.video_capture.isOpened():
            raise ValueError("Error: Could not open video.")

    def detect_license_plate(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = self.plate_detector.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(25, 25))
        
        for (x, y, w, h) in plates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            frame[y:y + h, x:x + w] = cv2.blur(frame[y:y + h, x:x + w], ksize=(10, 10))
            cv2.putText(frame, 'License Plate', (x - 3, y - 3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
        
        return frame

    def run(self):
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                print("Error: Failed to read video frame.")
                break

            processed_frame = self.detect_license_plate(frame)
            cv2.imshow('Video', processed_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = LicensePlateDetector(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml", 'vid.mp4')
    detector.run()

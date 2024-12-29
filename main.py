import cv2
import pytesseract
import numpy as np
import re

# Path for the Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class PlateRecognition:
    def __init__(self): 
        # Load the pre-trained Haar cascade xml file for plate detection model
        self.plate_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        )

    def enhance_image(self, input_image):
        """
        Enhance the input image for plate detection.
        Steps:
        - Convert the image to grayscale.
        - Apply a Filter to reduce noise while preserving edges.
        - Enhance the image using adaptive histogram equalization.
        """
        if input_image is None:
            print("Error: Image could not be loaded.")
            return None
        
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(blurred_image)
        
        return enhanced_image

    def enhance_plate(self, plate_image):
        """
        Preprocess the extracted plate region to improve OCR accuracy.
        Steps:
        - Resize the image if it is too small.
        - Convert it to grayscale.
        - Apply binary thresholding for better contrast.
        - Clean the image using morphological operations.
        """
        min_width = 300
        if plate_image.shape[1] < min_width:
            aspect_ratio = plate_image.shape[0] / plate_image.shape[1]
            new_width = min_width
            new_height = int(aspect_ratio * new_width)
            plate_image = cv2.resize(plate_image, (new_width, new_height))

        gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if len(plate_image.shape) == 3 else plate_image
        thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        kernel = np.ones((3, 3), np.uint8)
        cleaned_plate = cv2.morphologyEx(thresh_plate, cv2.MORPH_CLOSE, kernel)
        cleaned_plate = cv2.morphologyEx(cleaned_plate, cv2.MORPH_OPEN, kernel)

        return cleaned_plate

    def clean_text(self, raw_text):
        """
        Validate and clean the OCR result.
        - Remove unwanted characters.
        - Replace common OCR misinterpretations.
        """
        raw_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        if len(raw_text) < 4:
            return ""
        raw_text = raw_text.replace('O', '0').replace('I', '1').replace('S', '5')
        return raw_text

    def extract_plate_text(self, plate_image):
        """
        Use Tesseract OCR to recognize text from the processed license plate image.
        - Preprocess the plate region.
        - Configure and run Tesseract OCR.
        - Validate the extracted text.
        """
        processed_plate = self.enhance_plate(plate_image)
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        try:
            plate_text = pytesseract.image_to_string(processed_plate, config=custom_config)
            return self.clean_text(plate_text)
        except Exception as e:
            print(f"OCR Error: {str(e)}")
            return ""

    def detect_plate_from_image(self, image_path):
        """
        Detect and recognize license plates in a static image.
        - Load and preprocess the image.
        - Use Haar cascade to detect plates.
        - Extract, process, and recognize each detected plate.
        """
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Could not read image.")
            return

        processed_image = self.enhance_image(img)
        if processed_image is None:
            return

        plates = self.plate_cascade.detectMultiScale(
            processed_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 20),
            maxSize=(300, 100)
        )

        for (x, y, w, h) in plates:
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)

            plate_image = img[y:y + h, x:x + w]
            plate_text = self.extract_plate_text(plate_image)

            if plate_text:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                font_scale, font_thickness = 0.7, 2
                text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                text_x, text_y = x, max(0, y - 10)
                cv2.rectangle(img, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), (0, 255, 0), -1)
                cv2.putText(img, plate_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

        cv2.imshow('Number Plate Recognition', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_plate_from_video(self, video_path):
        """
        Detect and recognize license plates in a video stream.
        - Process each frame of the video.
        - Detect plates and recognize them in real-time.
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print("Error: Could not open video.")
            return

        while True:
            ret, frame = video.read()
            if not ret:
                break

            processed_frame = self.enhance_image(frame)
            if processed_frame is None:
                continue

            plates = self.plate_cascade.detectMultiScale(
                processed_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 20),
                maxSize=(300, 100)
            )

            for (x, y, w, h) in plates:
                plate_image = frame[y:y + h, x:x + w]
                plate_text = self.extract_plate_text(plate_image)

                if plate_text:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Video Plate Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = PlateRecognition()
    
    detector.detect_plate_from_image('images/Cars426.png') 
    detector.detect_plate_from_video('vid.mp4')

import os
import cv2
import numpy as np
import pytesseract
import tensorflow as tf
from tensorflow.keras.applications import VGG19

class OCRProcessor:
    def __init__(self):
        self.text_detector = self.load_text_detector()
        self.ocr_model = self.load_ocr_model()

    def load_text_detector(self):
        east_model_path = "models/frozen_east_text_detection.pb"
        net = cv2.dnn.readNet(east_model_path)
        return net

    def load_ocr_model(self):
        pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
        return pytesseract

    def enhance_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def detect_text(self, image, save_path="output.jpg"):
        orig = image.copy()
        (H, W) = image.shape[:2]
        
        newW, newH = (320, 320)
        rW = W / float(newW)
        rH = H / float(newH)
        image = cv2.resize(image, (newW, newH))
        
        blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.text_detector.setInput(blob)
        scores, geometry = self.text_detector.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
        
        rects, confidences = [], []
        for i in range(scores.shape[2]):
            for j in range(scores.shape[3]):
                if scores[0, 0, i, j] > 0.5:
                    offsetX, offsetY = j * 4.0, i * 4.0
                    angle = geometry[0, 4, i, j]
                    h, w = geometry[0, 0, i, j], geometry[0, 1, i, j]
                    endX = int(offsetX + (np.cos(angle) * w) + (np.sin(angle) * h))
                    endY = int(offsetY - (np.sin(angle) * w) + (np.cos(angle) * h))
                    startX, startY = int(endX - w), int(endY - h)
                    rects.append((startX, startY, endX, endY))
                    confidences.append(scores[0, 0, i, j])
        
        boxes = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)
        results = []

        if len(boxes) > 0 and isinstance(boxes[0], (list, np.ndarray)):  
            for i in range(len(boxes)):
                x1, y1, x2, y2 = rects[boxes[i][0]]
                results.append((x1, y1, x2, y2))
                cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            for i in range(len(boxes)):  # If it's a flat array
                x1, y1, x2, y2 = rects[boxes[i]]
                results.append((x1, y1, x2, y2))
                cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(save_path, orig)
        return results

    def extract_text(self, image):
        return self.ocr_model.image_to_string(image, lang='eng')

    def process_image(self, image_path, save_path="output.jpg"):
        image = cv2.imread(image_path)
        enhanced = self.enhance_image(image)
        text_regions = self.detect_text(image, save_path)
        extracted_text = self.extract_text(enhanced)
        return extracted_text, text_regions

# Example usage
if __name__ == "__main__":
    files = [os.path.join(root, file) for root, _, files in os.walk('images/test_images') for file in files]
    print(files)
    ocr = OCRProcessor()
    text, regions = ocr.process_image(files[0], "images\\output\\marked_sample.png")
    print("Extracted Text:", text)

from main import OCRProject
import logging

# Disable all logs
logging.getLogger('ppocr').setLevel(logging.WARNING)

images = ["test_images/sample1.png", "test_images/sample2.png"]

# Initialize the OCR project
ocr_project = OCRProject()
# Recognize text from an image (uses fine-tuned model by default)
ocr_project.recognize_text(image_path=images[0])
ocr_project.recognize_text(image_path=images[1])

# # Example: Fine-tune the model (uses default paths)
# ocr_project.fine_tune_model()

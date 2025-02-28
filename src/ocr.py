from paddleocr import PaddleOCR

class OCR:
    def __init__(self, model_dir=None, lang='en'):
        """
        Initialize the OCR model.
        :param model_dir: Path to the fine-tuned model directory (optional).
        :param lang: Language of the text (default is English).
        """
        self.model = PaddleOCR(use_angle_cls=True, lang=lang, rec_model_dir=model_dir)

    def recognize_text(self, image_path):
        """
        Recognize text from an image.
        :param image_path: Path to the input image.
        :return: List of recognized text.
        """
        result = self.model.ocr(image_path, cls=True)
        recognized_text = []
        for line in result:
            for word in line:
                recognized_text.append(word[1][0])  # Extract recognized text
        return recognized_text

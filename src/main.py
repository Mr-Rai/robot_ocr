import os
from ocr import OCR
from finetune import FineTuner
from robot.libraries.BuiltIn import BuiltIn

class OCRProject:
    def __init__(self):
        """
        Initialize the OCR project with default paths.
        """
        # Default paths
        self.default_model_dir = "models/fine_tuned/"
        self.default_pretrained_model_dir = "models/pretrained/rec_mv3_none_bilstm_ctc_v2.0_train/"
        self.default_output_dir = "models/fine_tuned/"
        self.default_config_file = "configs/rec/rec_mv3_none_bilstm_ctc.yml"
        self.default_data_dir = "data/images/"
        self.default_label_file = "data/train.txt"

    def recognize_text(self, image_path, model_dir=None):
        """
        Recognize text from an image.
        :param image_path: Path to the input image.
        :param model_dir: Path to the model directory (optional, defaults to fine-tuned model).
        """
        # Use fine-tuned model by default unless another model is specified
        if model_dir is None:
            model_dir = self.default_model_dir

        self.ocr = OCR(model_dir=model_dir)
        recognized_text = self.ocr.recognize_text(image_path)
        return recognized_text

    def fine_tune_model(self, data_dir=None, label_file=None, pretrained_model_dir=None, output_dir=None, config_file=None):
        """
        Fine-tune the model on a custom dataset.
        :param data_dir: Directory containing images and labels (optional, defaults to data/images/).
        :param label_file: Path to the label file (optional, defaults to data/train.txt).
        :param pretrained_model_dir: Path to the pre-trained model directory (optional, defaults to models/pretrained/rec_mv3_none_bilstm_ctc_v2.0_train/).
        :param output_dir: Directory to save the fine-tuned model (optional, defaults to models/fine_tuned/).
        :param config_file: Path to the training configuration file (optional, defaults to configs/rec/rec_mv3_none_bilstm_ctc.yml).
        """
        # Set default values if arguments are not provided
        if data_dir is None:
            data_dir = self.default_data_dir
        if label_file is None:
            label_file = self.default_label_file
        if pretrained_model_dir is None:
            pretrained_model_dir = self.default_pretrained_model_dir
        if output_dir is None:
            output_dir = self.default_output_dir
        if config_file is None:
            config_file = self.default_config_file

        self.fine_tuner = FineTuner(pretrained_model_dir, output_dir)
        self.fine_tuner.prepare_dataset(data_dir, label_file)
        self.fine_tuner.fine_tune(config_file)
        print(f"Fine-tuned model saved to {output_dir}")

def recognize_text(image_path, model_dir=None):
    try:
        text = OCRProject().recognize_text(image_path=image_path, model_dir=None)
        if isinstance(text, list):
            text = ' '.join(text)
            return text
    except Exception as e:
        BuiltIn().log(level='WARN', message=f'\nFailed to Process Image: ${e}', console=True)
        return 'Not Found'

def fine_tune_model():
    OCRProject().fine_tune_model()
    return True

import os
import time
from paddleocr import PaddleOCR, paddleocr
from robot.libraries.BuiltIn import BuiltIn

class FineTuner:
    def __init__(self, pretrained_model_dir, output_dir):
        """
        Initialize the fine-tuning process.
        :param pretrained_model_dir: Path to the pre-trained model directory.
        :param output_dir: Directory to save the fine-tuned model.
        """
        self.pretrained_model_dir = pretrained_model_dir
        self.output_dir = output_dir

    def prepare_dataset(self, data_dir, label_file):
        """
        Prepare the dataset for fine-tuning.
        :param data_dir: Directory containing images and labels.
        :param label_file: Path to the label file (e.g., train.txt).
        """
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Prepare the dataset (PaddleOCR expects a specific format)
        # You can add more dataset preprocessing logic here if needed
        print(f"Dataset prepared from {data_dir}")

    def fine_tune(self, config_file):
        """
        Fine-tune the model.
        :param config_file: Path to the configuration file for training.
        """
        paddleocr.train(
            config=config_file,
            pretrained_model=self.pretrained_model_dir,
            save_model_dir=self.output_dir
        )

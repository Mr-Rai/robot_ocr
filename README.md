# Enhanced OCR Application with Robot Framework wrapper

### 1. What is it about
The project is aimed at the areas of test automation, image processing and machine
learning. It seeks to improve the Optical Character Recognition (OCR) technology in the
Robot Framework with a view to advancing the automation of industrial testing tasks. The
focus is on solving the problems of noisy text, font and language diversity, and the
performance of the system in real time conditions.

-------

### 2. Objectives
- Enhance existing OCR systems that are capable of working on noisy, multilingual
sources in different font styles more reliably and more precisely than they do currently.
- Implement an OCR system as a library of the Robot Framework where the OCR system
is of a suitable design to allow users to Convert OCR into a global standard.
- Provide further image enhancement techniques such as de-noising, skew removal and
region of interest to improve the overall performance of the proposed OCR. • Allow the
data to be processed on real time basis to improve the speed and efficiency in the
recognition and verification of text during automated testing.
- Encourage the community to collaborate by releasing the tool on GitHub with
appropriate descriptions, manuals, and examples.

-------
## The project has two main functionalities:
- **Text Recognition**: Recognize text from images using a pre-trained or fine-tuned model.
- **Fine-Tuning**: Allow users to fine-tune the model on their custom dataset.

--------

### Directory Structure
```
robot_ocr/
│
├── configs/
|   ├── coordinates.json        # Coordinate file for storing region of interest
|   ├── rec/                    # YML file for configuration of model training
|       ├── rec_mv3_none_bilstm_ctc.yml
|
├── data/                       # Folder for custom dataset
│   ├── images/                 # Folder for training images
│   ├── labels/                 # Folder for label files (e.g., image1.txt, image2.txt)
│   └── train.txt               # File mapping images to labels (e.g., image1.jpg\tHello)
│
├── models/                     # Folder to store pre-trained and fine-tuned models
│   ├── pretrained/             # Pre-trained models
│   └── fine_tuned/             # Fine-tuned models
│
├── src/                        # Python functions
│   └── finetune.py             # Python code for handling model finetuning
│   └── main.py                 # Main function to interact with robot
│   └── ocr.py                  # Python code for recognizing text
│   └── utils.py                # Utils for robot keywords
│
├── test_images/                # Folder for test images
│   └── cropped_images/         # Automatically created by robot keyword
│   └── enhanced_images/        # Automatically created by robot keyword
|
├── unit_test/                  # Folder for unit test files
│   └── unit_test_ocr.robot     # Robot unit test for text Recognition
│   └── unit_test_train.robot   # Robot unit test for finetuning model
|
├── ReadMe.md                   # Read me file
├── requriements.txt            # Requirements file for python dependencies
|
```
--------

### Code Implementation
1. **ocr.py** (Text Recognition Class): This class will handle loading the model and recognizing text from images.
2. **finetune.py** (Fine-Tuning Class): This class will handle fine-tuning the model on a custom dataset.
3. **main.py** (Main Script): This script will provide a user-friendly interface to recognize text or fine-tune the model.
----------------------------------------

## User Guide: How to Create Training Data
1. **Dataset Structure**
    - Create a folder named data/ in your project directory.
    - Inside data/, create two subfolders:
    - images/: Store all training images (e.g., image1.jpg, image2.jpg).
    - labels/: Store corresponding label files (e.g., image1.txt, image2.txt).

2. **Label Files**
    - For each image, create a .txt file with the same name as the image.
    - The .txt file should contain the ground truth text for the image.
    - Example:
        - image1.jpg → image1.txt (content: Hello World)
        - image2.jpg → image2.txt (content: OCR is fun)

3. **Training Data File (train.txt)**
    - Create a file named train.txt in the data/ folder.
    - Each line in train.txt should map an image to its label file.
    - Format:
```
data/images/image1.jpg	data/labels/image1.txt
data/images/image2.jpg	data/labels/image2.txt
```

4. **Example Dataset**
```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
├── labels/
│   ├── image1.txt
│   ├── image2.txt
└── train.txt
```
--------------
## How to Use the Project

1. **Recognize Text**
```
from main import OCRProject

# Initialize the OCR project
ocr_project = OCRProject()

# Recognize text from an image (uses fine-tuned model by default)
ocr_project.recognize_text(image_path="data/images/test_image.jpg")
```

2. **Fine-Tune the Model**
```
from main import OCRProject

# Initialize the OCR project
ocr_project = OCRProject()

# Fine-tune the model (uses default paths)
ocr_project.fine_tune_model()
```

3. **How to Use the YAML File**: Place the YAML File:
   - Modify Paths in YAML:
   - Update the pretrained_model path to point to your downloaded pre-trained model.
   - Update data_dir and label_file_list if your dataset is in a different location.
   - Use in Fine-Tuning: The fine_tune_model method in main.py will use this YAML file by default.


import os
import time
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from imutils.object_detection import non_max_suppression
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

SAVED_MODEL_PATH = 'models\\frozen_east_text_detection.pb'
SAVED_MODEL_PATH = 'https://tfhub.dev/captain-pool/esrgan-tf2/1'

model = hub.load(SAVED_MODEL_PATH)
folder_path = 'images\\test_images'
pipeline_dict = {}
os.listdir(folder_path)

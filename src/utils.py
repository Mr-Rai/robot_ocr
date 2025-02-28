import cv2
import os
import base64
import pandas as pd
import numpy as np
import pytesseract
import json
from robot.libraries.BuiltIn import BuiltIn
from robot.api.deco import keyword

CROPPED_IMG_PATH = '../test_images/cropped_images/'
ENHANCED_IMG_PATH = '../test_images/enhanced_images/'
COORDINATE_CONFIG_PATH = '../configs/coordinates.json'

[os.mkdir(dir) for dir in [
    CROPPED_IMG_PATH, ENHANCED_IMG_PATH, COORDINATE_CONFIG_PATH] if not os.path.exists(dir)]

class utils:
    def __init__(self):
        # Create dir if not present
        required_dir = [CROPPED_IMG_PATH, ENHANCED_IMG_PATH]
        for dir in required_dir:
            try:
                os.mkdir(dir)
            except:
                continue

    def log_html_table(self, list_data):
        df = pd.DataFrame(list_data, columns=list_data[0])

        # Convert DataFrame to HTML
        html_table = df.to_html(escape=False)

        with open("image_comparison.html", "w") as f:
            f.write(html_table)

    def enhance_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Apply thresholding to binarize the image
        _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Invert the image to make the text black and background white
        inverted = cv2.bitwise_not(binary)

        kernel = np.ones((3,3),np.uint8)
        img = cv2.erode(inverted,kernel,iterations = 1)
        # Save enhanced images
        if '\\' in img_path:
            img_name = img_path.split('\\')[-1]
        else:
            img_name = img_path.split('/')[-1]
        cv2.imwrite(ENHANCED_IMG_PATH+img_name, img)
        return ENHANCED_IMG_PATH+img_name

    # def enhance_image(self, image_path):
    #     """Enhance an image for OCR (brightness, contrast, noise reduction, sharpening)."""
    #     # Read the image
    #     image = cv2.imread(image_path)

    #     # Convert the image to grayscale for better OCR performance
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #     # Analyze the image to adjust brightness
    #     mean_brightness = np.mean(image)  # Calculate the average pixel brightness

    #     # Automatically adjust brightness based on the image's brightness
    #     if mean_brightness < 100:  # Image is too dark
    #         brightness_delta = 30
    #         print("Image is too dark. Increasing brightness.")
    #     elif mean_brightness > 180:  # Image is too bright
    #         brightness_delta = -30
    #         print("Image is too bright. Decreasing brightness.")
    #     else:
    #         brightness_delta = 0
    #         print("Image brightness is optimal.")

    #     # Adjust brightness using a linear transformation
    #     image = cv2.convertScaleAbs(image, alpha=1, beta=brightness_delta)

    #     # # Increase contrast by applying histogram equalization
    #     # gray = cv2.equalizeHist(gray)  # Perform histogram equalization to improve contrast

    #     # # Apply Gaussian blur to reduce noise (helps in OCR)
    #     # image = cv2.GaussianBlur(image, (5, 5), 0)

    #     # Use adaptive thresholding to create a binary image (improves clarity of text)
    #     # Adaptive thresholding helps when the lighting varies across the image
    #     image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                 cv2.THRESH_BINARY, 11, 2)

    #     # # Sharpen the image (to make text crisper)
    #     # kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    #     # image = cv2.filter2D(image, -1, kernel)

    #     # Save enhanced images
    #     if '\\' in image_path:
    #         image_name = image_path.split('\\')[-1]
    #     else:
    #         image_name = image_path.split('/')[-1]
    #     cv2.imwrite(ENHANCED_IMG_PATH+image_name, image)
    #     return ENHANCED_IMG_PATH+image_name

    def get_curwd(self):
        curdir = os.getcwd()
        files = os.listdir(curdir+'../')
        return  curdir, files

    def read_text_from_image(self, image_path):
        """Read text from an image file"""
        subimage = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        height, _= subimage.shape
        scale_factor = 90 / int(height)
        subimage = cv2.resize(subimage, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)
        # subimage = self.enhance_image(subimage)
        whitelist_chars = " *@#$%^&()+-=[]{}|;:,.<>?/ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890"
        config = f"--psm 6 --oem 3 -c tessedit_char_whitelist={whitelist_chars}"
        text = pytesseract.image_to_string(subimage, config=config)
        text = text.replace('\n', ' ')
        return text

    def crop_image_region(self, image_path, region_name=None, x1=0, y1=0, x2=0, y2=0):
        if x1 == 0 and x2 == 0 and y1 == 0 and y2 == 0 and region_name is None:
            return image_path
        image_file = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if region_name:
            with open(COORDINATE_CONFIG_PATH, 'r') as file:
                region_data = json.load(file)
                try:
                    region_data = region_data['ROI'][region_name]
                    x1, y1, x2, y2 = region_data['x1'], region_data['y1'], region_data['x2'], region_data['y2']
                except Exception as e:
                    BuiltIn().log(message=f"\nRegion name {region_name} not found in the config file. Taking full Image", level='WARN', console=True)
                    x1, y1, x2, y2 = 0, 0, image_file.shape[0], image_file.shape[1]
            if '\\' in image_path:
                img_name = image_path.split('\\')[-1]
            else:
                img_name = image_path.split('/')[-1]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            try:
                cropped_image_file = image_file[y1:y2, x1:x2] 
            except Exception as e:
                BuiltIn().log(message=f"\nError cropping image: {e}, Taking full image", level='WARN', console=True)
                cropped_image_file = image_file[0:image_file.shape[1], 0:image_file.shape[0]]
            cv2.imwrite(f'{CROPPED_IMG_PATH}{region_name}_{img_name}', cropped_image_file)
        return CROPPED_IMG_PATH+region_name+'_'+img_name

    def log_image(self, img_path, width=500):
        with open(img_path, 'rb') as image_file:
            b64_img_str = base64.b64encode(image_file.read()).decode('utf-8')
        img_html = f"<img src='data:image/png;base64, {str(b64_img_str)}' width='{int(width)}px' alt='Image Unavailable'/>"
        BuiltIn().log(img_html, html=True)
        return  img_html

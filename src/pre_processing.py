import cv2
import os
import pandas as pd
import numpy as np
import pytesseract
from robot.libraries.BuiltIn import BuiltIn

CROPPED_IMG_PATH = 'images\\cropped_images\\'
ENHANCED_IMG_PATH = 'images\\enhanced_images\\'

def create_html_table(list_data):
    df = pd.DataFrame(list_data, columns=['image_name', 'image', 'cropped_image', 'enhanced_image'])

    # Convert DataFrame to HTML
    html_table = df.to_html(escape=False)

    with open("image_comparison.html", "w") as f:
        f.write(html_table)


def enhance_image(img):
    # Apply thresholding to binarize the image
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Invert the image to make the text black and background white
    inverted = cv2.bitwise_not(binary)

    kernel = np.ones((3,3),np.uint8)
    img = cv2.erode(inverted,kernel,iterations = 1)

    return inverted

def read_text_from_image(image_path):
    """Read text from an image file"""
    subimage = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, _= subimage.shape
    scale_factor = 90 / int(height)
    subimage = cv2.resize(subimage, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)
    subimage = enhance_image(subimage)
    whitelist_chars = " *@#$%^&()+-=[]{}|;:,.<>?/ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890"
    config = f"--psm 6 --oem 3 -c tessedit_char_whitelist={whitelist_chars}"
    text = pytesseract.image_to_string(subimage, config=config)
    text = text.replace('\n', ' ')
    # Save enhanced images
    img_name = image_path.split('\\')[-1]
    cv2.imwrite(ENHANCED_IMG_PATH+img_name, subimage)
    return text, ENHANCED_IMG_PATH+img_name

def crop_region(image_path, x1, y1, x2, y2, region_name):
    img_name = image_path.split('\\')[-1]
    image_file = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cropped_image_file = image_file[y1:y2, x1:x2] 
    cv2.imwrite(f'{CROPPED_IMG_PATH}{region_name}_{img_name}', cropped_image_file)
    return CROPPED_IMG_PATH+region_name+'_'+img_name

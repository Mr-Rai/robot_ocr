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

# def display_image(img):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()

#TODO: DOWNLOAD AND KEEP IMAGES HERE
# from google.colab import files
# uploaded = files.upload()
# for filename in uploaded.keys():
#   print('User uploaded file "{name}" with {length} bytes'.format(name=filename, length=len(uploaded[filename])))

# zip_file_path = '/content/test_images.zip'
# extraction_directory = '/content/test_images/'

# unzip "{zip_file_path}" -d "{extraction_directory}"

#TODO: Use locally downloaded model
# SAVED_MODEL_PATH = 'https://tfhub.dev/captain-pool/esrgan-tf2/1'
SAVED_MODEL_PATH = 'models\\frozen_east_text_detection.pb'

def post_process(orig, scores, geometry, confThreshold=0.5, nmsThreshold=0.4, rW=1, rH=1):
    """
    Post-processes the output of the EAST text detector model to suppress weak, overlapping bounding boxes.
    
    Args:
        orig (numpy.ndarray): The original image.
        scores (numpy.ndarray): The scores from the EAST detector.
        geometry (numpy.ndarray): The geometry from the EAST detector.
        confThreshold (float): Confidence threshold to filter weak detections.
        nmsThreshold (float): Non-maxima suppression threshold.
        rW (float): Width ratio.
        rH (float): Height ratio.

    Returns:
        masked_image (numpy.ndarray): The resulting image with bounding boxes applied.
    """
    (rects, confidences) = decode_predictions(scores, geometry, confThreshold)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences, overlapThresh=nmsThreshold)


    mask = np.zeros(orig.shape[:2], dtype='uint8')

    padding = 10  # Adjust padding as needed

    for (startX, startY, endX, endY) in boxes:
        # Apply padding around each bounding box and ensure the coordinates are within image bounds
        p_startX = max(0, int(startX * rW ) - padding )
        p_startY = max(0, int(startY * rH ) - padding)
        p_endX = min(orig.shape[1] -1 , int( endX * rW ) + padding )
        p_endY = min(orig.shape[0] -1 , int( endY * rH ) + padding )

        # Draw white rectangles on the mask for each bounding box (with padding)
        cv2.rectangle(mask, (p_startX, p_startY), (p_endX, p_endY), 255, -1)


    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(orig, orig, mask=mask)

    return masked_image

def decode_predictions(scores, geometry, confThreshold=0.5):
    """
    Decodes the predictions from the EAST text detector model.

    Args:
        scores (numpy.ndarray): The scores from the EAST detector.
        geometry (numpy.ndarray): The geometry from the EAST detector.
        confThreshold (float): Confidence threshold to filter weak detections.

    Returns:
        tuple: A tuple containing the bounding boxes and associated confidences.
    """
    # Grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # Loop over the number of rows
    for y in range(0, numRows):
        # Extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # Loop over the number of columns
        for x in range(0, numCols):
            # If our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < confThreshold:
                continue

            # Compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # Extract the rotation angle for the prediction and
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # Compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # Add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # Return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)

##CNN Model for applying masking around the text
def detect_text_east(image_path):
    """
    Detects text in an image using the EAST text detector model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        result_image (numpy.ndarray): The resulting image with text bounding boxes applied.
    """
    # Load image
    image = cv2.imread(image_path)
    orig = image.copy()
    (Ori_H, Ori_W) = image.shape[:2]

    # Set the new width and height and then determine the ratio in change
    (newW, newH) = (320, 320)
    rW = Ori_W / float(newW)
    rH = Ori_H / float(newH)

    # Resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # Define the two output layer names for the EAST detector model
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    # Load the pre-trained EAST text detector
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    # Construct a blob from the image and then perform a forward pass
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    rW = Ori_W / float(newW)
    rH = Ori_H / float(newH)

    # Decode the predictions, then  apply non-maxima suppression to suppress weak, overlapping bounding boxes
    result_image = post_process(orig.copy(), scores, geometry, confThreshold=0.5, nmsThreshold=0.4, rW=rW, rH=rH)
    # Return the original image (for now)
    return result_image

def preprocess_image(image_path):
    """Loads image from path and preprocesses to make it model ready.
    
    Args:
        image_path (str): Path to the image file.

    Returns:
        tf.Tensor: Preprocessed image tensor.
    """
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
    """
    Saves unscaled Tensor Images.

    Args:
        image (tf.Tensor): 3D image tensor. [height, width, channels]
        filename (str): Name of the file to save.
    """
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(filename)

# %matplotlib inline
def plot_image(image, title=""):
    """
    Plots images from image tensors.

    Args:
        image (tf.Tensor): 3D image tensor. [height, width, channels].
        title (str): Title to display in the plot.
    """
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)

model = tf.saved_model.load(SAVED_MODEL_PATH)
# model = hub.load(SAVED_MODEL_PATH)
folder_path = 'images\\test_images'
# os.mkdir('/images/output/')
pipeline_dict = {}
os.listdir(folder_path)

for image_file_name in os.listdir(folder_path):
    image_name = image_file_name.split('.')[0]
    image_path = folder_path + image_file_name

    print(f'Image Name: {image_name}')

    import sys
    sys.exit('BYE BYE !!')

    if image_name is None or image_name == '':
      continue

    print(f'IMAGE NAME: ', image_name)
    start = time.time()

    result_image = detect_text_east(image_path)
    #display_image(result_image)
    output_path = f'/images/output/{image_name}_stage_1.jpg'
    status = cv2.imwrite(output_path, result_image)

    ## SUPER RESOLUTION
    IMAGE_PATH = f'/images/output/{image_name}_stage_1.jpg'
    hr_image = preprocess_image(IMAGE_PATH)
    #plot_image(tf.squeeze(hr_image), title="Original Image")

    fake_image = model(hr_image)
    fake_image = tf.squeeze(fake_image)

    #plot_image(tf.squeeze(fake_image), title="Super Resolution")
    save_image(tf.squeeze(fake_image), filename=f"/images/output/{image_name}_stage_2.jpg")

    ##OCR

    masked_image_path = f'/images/output/{image_name}_stage_2.jpg'
    #masked_image_path = '/content/test_images/00bedd5c2fbf2dff.jpg'

    # Read the image using OpenCV (PyTesseract also works with PIL images)
    masked_image = cv2.imread(masked_image_path)

    # Convert the image to RGB (PyTesseract expects images in RGB format)
    masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

    # Use PyTesseract to extract text
    extracted_text = pytesseract.image_to_string(masked_image_rgb)

    words = extracted_text.split()
    words = [re.sub(r'[^\w\s]', '', word) for word in words]


    print("Time Taken: %f" % (time.time() - start))

    print('==================================')
    print("Extracted Text:")
    print(words)
    pipeline_dict[image_name] = words
    print('===================================')

tesseract_dict = {}

for image_file_name in os.listdir(folder_path):
    start = time.time()

    image_name = image_file_name.split('.')[0]
    image_path = folder_path + image_file_name

    if image_name is None or image_name == '':
      continue
    masked_image_path = image_path
    #masked_image_path = '/content/test_images/00bedd5c2fbf2dff.jpg'

    # Read the image using OpenCV (PyTesseract also works with PIL images)
    masked_image = cv2.imread(masked_image_path)

    # Convert the image to RGB (PyTesseract expects images in RGB format)
    masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

    # Use PyTesseract to extract text
    extracted_text = pytesseract.image_to_string(masked_image_rgb)

    words = extracted_text.split()
    words = [re.sub(r'[^\w\s]', '', word) for word in words]

    print("Time Taken: %f" % (time.time() - start))

    print('==================================')
    print("Extracted Text:")
    print(words)
    print('===================================')

    tesseract_dict[image_name] = words
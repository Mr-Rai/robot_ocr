import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Step 1: Load the Object Detection Model
def load_model(model_url):
    print("Loading model from TensorFlow Hub...")
    model = hub.load(model_url)
    print("Model loaded successfully!")
    return model

# Step 2: Preprocess the Input Image
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to float32 and normalize
    input_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.float32) / 255.0

    # Add batch dimension
    input_tensor = tf.expand_dims(input_tensor, axis=0)

    return input_tensor, image_rgb

# Step 3: Extract and Postprocess Results
def extract_results(detections, image_shape, score_threshold=0.5):
    height, width, _ = image_shape

    boxes = detections["detection_boxes"].numpy()[0]
    scores = detections["detection_scores"].numpy()[0]
    classes = detections["detection_classes"].numpy()[0].astype(int)

    results = []
    for box, score, class_id in zip(boxes, scores, classes):
        if score > score_threshold:
            ymin, xmin, ymax, xmax = box
            results.append({
                "box": [int(ymin * height), int(xmin * width), int(ymax * height), int(xmax * width)],
                "score": score,
                "class": class_id
            })
    return results

# Step 4: Visualize Results
def draw_detections(image, detections):
    for detection in detections:
        ymin, xmin, ymax, xmax = detection["box"]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f"Class: {detection['class']}, Score: {detection['score']:.2f}"
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# Step 5: Run the Complete Detection Pipeline
def run_object_detection(model_url, image_path, score_threshold=0.5):
    # Preprocess the image
    input_tensor, original_image = preprocess_image(image_path)

    # Load model
    model = load_model(model_url)

    # Run inference
    print("Running inference...")
    detections = model(input_tensor)

    # Extract results
    detections_processed = extract_results(detections, original_image.shape, score_threshold)

    # Visualize results
    print("Visualizing results...")
    draw_detections(original_image.copy(), detections_processed)

folder_path = 'images\\test_images'
# Example usage
MODEL_URL = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"  # Replace with your desired model URL
IMAGE_PATH = f'{folder_path}\\test1.png'  # Replace with the path to your image
run_object_detection(MODEL_URL, IMAGE_PATH)

# image = cv2.imread(IMAGE_PATH)
# cv2.imshow('Image Window', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

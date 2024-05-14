import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Function to detect fire in an image
def detect_fire(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Resize image to match model input size
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict the probability of each class
    preds = model.predict(x)

    # Decode predictions
    decoded_preds = decode_predictions(preds, top=1)[0]

    # Check if the top prediction is related to fire
    for pred in decoded_preds:
        if 'fire' in pred[1]:
            return True  # Fire detected

    return False  # No fire detected

# Example usage
image_path = "test1.png"
if detect_fire(image_path):
    print("Fire detected in the image.")
else:
    print("No fire detected in the image.")

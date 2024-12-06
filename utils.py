import cv2
import numpy as np
from skimage.transform import resize
import pickle
import os

# Constants
EMPTY = True
NOT_EMPTY = False
MODEL_PATH = "model/model.p"
VIDEO_PATH = "videos/"

# Load pre-trained model if available
if os.path.exists(MODEL_PATH):
    MODEL = pickle.load(open(MODEL_PATH, "rb"))
else:
    MODEL = None

# Initialize the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

def create_mask(frame, use_bg_subtraction=True):
    """Create a binary mask for parking spots."""
    if use_bg_subtraction:
        # Use background subtraction to detect movement
        fgmask = fgbg.apply(frame)  # Apply the background subtractor
        fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
        return fgmask
    else:
        # Fallback to thresholding method
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        return mask

def preprocess_spot(spot_bgr):
    """Resize and preprocess the image to match the model's input."""
    spot_resized = resize(spot_bgr, (15, 15, 3))
    return spot_resized.flatten()

def extract_parking_spots(frame, mask):
    """Extract parking spots' bounding boxes using contours."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spots = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        spots.append([x, y, w, h])
    return spots

def empty_or_not(spot_bgr):
    """Predict whether the parking spot is empty or not."""
    # Check if model is loaded, if not load it
    global MODEL  # Ensure we're using the global variable

    if MODEL is None:
        # Load the model if it's not already loaded
        if os.path.exists(MODEL_PATH):
            MODEL = pickle.load(open(MODEL_PATH, "rb"))
        else:
            raise ValueError("Model is not trained yet.")

    flat_data = preprocess_spot(spot_bgr)
    y_output = MODEL.predict([flat_data])
    return EMPTY if y_output == 0 else NOT_EMPTY


def train_model(labeled_data):
    """Train a model with labeled data."""
    from sklearn.ensemble import RandomForestClassifier
    X = []
    y = []

    for spot_image, label in labeled_data:
        spot_data = preprocess_spot(spot_image)
        X.append(spot_data)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    # Ensure the directory exists before saving the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Save the model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model


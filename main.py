import cv2
import os
from utils import create_mask, extract_parking_spots, empty_or_not, train_model

VIDEO_PATH = "data/"
OUTPUT_PATH = "output/"
FRAME_SIZE = (640, 480)  # Resize frame size to reduce computational cost
FRAME_SAMPLE_RATE = 10  # Process every 10th frame to reduce training time

# Initialize background subtractor (for detecting changes)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

def process_video(video_file):
    """Process video and detect empty/occupied spots."""
    # Read the video file
    cap = cv2.VideoCapture(VIDEO_PATH + video_file)

    # Create the output folder
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    frame_count = 0
    labeled_data = []  # Store labeled data for training
    model_trained = False  # Flag to check if the model is trained
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process only every 'FRAME_SAMPLE_RATE'-th frame to reduce training time
        if frame_count % FRAME_SAMPLE_RATE != 0:
            frame_count += 1
            continue
        
        # Resize frame
        frame_resized = cv2.resize(frame, FRAME_SIZE)

        # Create mask for the current frame
        mask = create_mask(frame_resized)
        
        # Extract parking spots
        spots = extract_parking_spots(frame_resized, mask)

        # Apply background subtraction to detect occupied spots
        fgmask = fgbg.apply(frame_resized)
        fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]  # Threshold to clean up mask
        
        # Process each parking spot automatically
        for spot in spots:
            x, y, w, h = spot
            spot_image = frame_resized[y:y+h, x:x+w]
            
            # Check if the parking spot has significant movement (occupied or empty)
            spot_mask = fgmask[y:y+h, x:x+w]
            non_zero_count = cv2.countNonZero(spot_mask)  # Count non-zero pixels in the mask

            # If non-zero pixels are detected, the spot is likely occupied
            label = 1 if non_zero_count > 500 else 0  # Threshold can be adjusted based on spot size

            labeled_data.append((spot_image, label))

        # Only call the model for predictions after collecting enough labeled data
        if labeled_data:
            # Train the model after labeling
            model = train_model(labeled_data)
            model_trained = True
            print("Model trained and saved.")
        
        # Once the model is trained, predict using the trained model
        if model_trained:
            for spot in spots:
                x, y, w, h = spot
                spot_image = frame_resized[y:y+h, x:x+w]
                
                # Predict using the trained model
                result = empty_or_not(spot_image)
                color = (0, 255, 0) if result == 0 else (0, 0, 255)
                cv2.rectangle(frame_resized, (x, y), (x+w, y+h), color, 2)
        
        # Show the frame with bounding boxes
        cv2.imshow("Parking Spot Detection", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Ensure model is trained before ending
    if labeled_data and model_trained:
        model = train_model(labeled_data)
        print("Model trained and saved.")
    else:
        print("No labeled data to train the model.")

if __name__ == "__main__":
    # Process the video and train the model
    video_file = "Busy Parking.mp4"  # Add your video file here
    process_video(video_file)

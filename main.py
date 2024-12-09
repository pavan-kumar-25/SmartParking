import cv2
import numpy as np
from sklearn.cluster import DBSCAN

class MaskBasedParkingDetector:
    def __init__(self):
        self.reference_mask = None
        self.parking_spaces = []
        self.background_model = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True)
    
    def create_reference_mask(self, frame):
        """Create binary mask from reference frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 19, 2
        )
        
        # Morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        self.reference_mask = mask
        return mask
    
    def find_parking_slots(self, mask):
        """Find parking slots using contour detection on the mask"""
        # Find contours in the mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area and shape
        slots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter based on area (adjust thresholds as needed)
            if 1000 < area < 20000:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio
                aspect_ratio = float(w) / h
                if 0.4 < aspect_ratio < 2.5:
                    slots.append((x, y, x + w, y + h))
        
        # Cluster nearby rectangles
        if slots:
            # Convert to numpy array for clustering
            points = np.array([[rect[0], rect[1]] for rect in slots])
            clustering = DBSCAN(eps=50, min_samples=2).fit(points)
            
            # Process clusters
            unique_labels = set(clustering.labels_)
            final_slots = []
            
            for label in unique_labels:
                if label != -1:  # Ignore noise
                    cluster_points = points[clustering.labels_ == label]
                    # Average the rectangles in each cluster
                    avg_x = np.mean([slots[i][0] for i in np.where(clustering.labels_ == label)[0]])
                    avg_y = np.mean([slots[i][1] for i in np.where(clustering.labels_ == label)[0]])
                    avg_w = np.mean([slots[i][2] - slots[i][0] for i in np.where(clustering.labels_ == label)[0]])
                    avg_h = np.mean([slots[i][3] - slots[i][1] for i in np.where(clustering.labels_ == label)[0]])
                    
                    final_slots.append((
                        int(avg_x), int(avg_y),
                        int(avg_x + avg_w), int(avg_y + avg_h)
                    ))
            
            self.parking_spaces = final_slots
            return final_slots
        
        return []
    
    def detect_occupancy(self, frame):
        """Detect occupancy using background subtraction and the reference mask"""
        # Apply background subtraction
        fg_mask = self.background_model.apply(frame)
        
        # Apply threshold to get binary mask
        _, thresh = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
        
        # Process each parking space
        frame_display = frame.copy()
        for x1, y1, x2, y2 in self.parking_spaces:
            # Extract ROI from current frame mask
            roi_mask = thresh[y1:y2, x1:x2]
            
            # Calculate occupancy based on white pixel ratio
            white_pixels = np.sum(roi_mask > 0)
            total_pixels = roi_mask.size
            occupancy_ratio = white_pixels / total_pixels
            
            # Determine if occupied (adjust threshold as needed)
            is_occupied = occupancy_ratio > 0.3
            
            # Draw results
            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
            status = "Occupied" if is_occupied else "Empty"
            cv2.putText(frame_display, status, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show occupancy ratio for debugging
            ratio_text = f"{occupancy_ratio:.2f}"
            cv2.putText(frame_display, ratio_text, (x1, y2+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame_display

def main():
    # Initialize detector
    detector = MaskBasedParkingDetector()
    
    # Open video file
    video_path = 'Busy Parking.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Read first frame for creating reference mask
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video frame")
        return
    
    # Create reference mask and find parking slots
    print("Creating reference mask...")
    mask = detector.create_reference_mask(frame)
    
    print("Finding parking slots...")
    slots = detector.find_parking_slots(mask)
    print(f"Found {len(slots)} parking slots")
    
    # Show reference mask and detected slots
    cv2.imshow('Reference Mask', mask)
    
    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and display results
        result_frame = detector.detect_occupancy(frame)
        
        cv2.imshow('Parking Detection', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
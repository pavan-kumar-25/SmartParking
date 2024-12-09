import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import threading
import queue
import os
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify

app = Flask(__name__)

# Global variables
video_path = None
processing_thread = None
frame_queue = queue.Queue(maxsize=10)
status_queue = queue.Queue(maxsize=10)
stop_thread = threading.Event()


class MaskBasedParkingDetector:
    def __init__(self):
        self.reference_mask = None
        self.parking_spaces = []
        self.background_model = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True)
        self.total_slots = 0
        self.occupied_slots = 0

    def create_reference_mask(self, frame):
        """Create binary mask from reference frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 19, 2
        )
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        self.reference_mask = mask
        return mask

    def find_parking_slots(self, mask):
        """Find parking slots using contour detection."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        slots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 20000:  # Adjust thresholds as needed
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.4 < aspect_ratio < 2.5:
                    slots.append((x, y, x + w, y + h))
        if slots:
            points = np.array([[rect[0], rect[1]] for rect in slots])
            clustering = DBSCAN(eps=50, min_samples=2).fit(points)
            unique_labels = set(clustering.labels_)
            final_slots = []
            for label in unique_labels:
                if label != -1:  # Ignore noise
                    indices = np.where(clustering.labels_ == label)[0]
                    cluster_slots = [slots[i] for i in indices]
                    avg_x = int(np.mean([rect[0] for rect in cluster_slots]))
                    avg_y = int(np.mean([rect[1] for rect in cluster_slots]))
                    avg_w = int(np.mean([rect[2] - rect[0] for rect in cluster_slots]))
                    avg_h = int(np.mean([rect[3] - rect[1] for rect in cluster_slots]))
                    final_slots.append((avg_x, avg_y, avg_x + avg_w, avg_y + avg_h))
            self.parking_spaces = final_slots
            return final_slots
        return []

    def detect_occupancy(self, frame):
        """Detect occupancy using background subtraction."""
        fg_mask = self.background_model.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
        frame_display = frame.copy()
        self.occupied_slots = 0
        for x1, y1, x2, y2 in self.parking_spaces:
            roi_mask = thresh[y1:y2, x1:x2]
            white_pixels = np.sum(roi_mask > 0)
            total_pixels = roi_mask.size
            occupancy_ratio = white_pixels / total_pixels
            is_occupied = occupancy_ratio > 0.3
            if is_occupied:
                self.occupied_slots += 1
            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
            status = "Occupied" if is_occupied else "Empty"
            cv2.putText(frame_display, status, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame_display

    def get_status(self):
        """Return current parking status."""
        return {
            'total_slots': len(self.parking_spaces),
            'occupied_slots': self.occupied_slots,
            'empty_slots': len(self.parking_spaces) - self.occupied_slots,
            'occupancy_percentage': round(
                (self.occupied_slots / len(self.parking_spaces) * 100) if self.parking_spaces else 0, 2)
        }


def process_video():
    global video_path
    detector = MaskBasedParkingDetector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return
    mask = detector.create_reference_mask(frame)
    detector.find_parking_slots(mask)
    while cap.isOpened() and not stop_thread.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        result_frame = detector.detect_occupancy(frame)
        _, buffer = cv2.imencode('.jpg', result_frame)
        if not frame_queue.full():
            frame_queue.put(buffer.tobytes())
        if not status_queue.full():
            status_queue.put(detector.get_status())
    cap.release()


def generate_frames():
    while not stop_thread.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    global video_path, processing_thread
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    video_path = f"uploads/video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    os.makedirs('uploads', exist_ok=True)
    video_file.save(video_path)
    if processing_thread and processing_thread.is_alive():
        stop_thread.set()
        processing_thread.join()
    stop_thread.clear()
    processing_thread = threading.Thread(target=process_video)
    processing_thread.start()
    return jsonify({'success': True})


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def get_status():
    if not status_queue.empty():
        return jsonify(status_queue.get())
    return jsonify({'total_slots': 0, 'occupied_slots': 0, 'empty_slots': 0, 'occupancy_percentage': 0})


if __name__ == '__main__':
    app.run(debug=True)

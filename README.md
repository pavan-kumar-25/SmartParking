SmartParking
SmartParking is a computer vision-based system designed to monitor and display real-time parking spot availability from video feeds. The project leverages OpenCV for image processing and a pre-trained Support Vector Classifier (SVC) model for determining spot occupancy.

Features
Real-Time Detection: Monitors a video feed and identifies available parking spots.
ML Integration: Uses a machine learning model to classify parking spots as empty or occupied.
User-Friendly Visualization: Displays status using green (available) and red (occupied) bounding boxes.
Scalable Solution: Works with parking areas of any size or configuration.
Repository Structure
bash
Copy code
SmartParking/
├── main.py                   # Main script for processing and monitoring
├── util.py                   # Utility functions for detection and preprocessing
├── mask_1920_1080.png        # Mask image defining parking spot locations
├── data/
│   └── parking_1920_1080.mp4 # Sample parking video for demonstration
├── model/
│   └── model.p               # Pre-trained machine learning model (SVC)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
Requirements
Python 3.8 or higher
Required Python libraries:
opencv-python
numpy
scikit-learn
Installation
Clone the Repository
bash
Copy code
git clone https://github.com/pavan-kumar-25/SmartParking.git
cd SmartParking
Install Dependencies
bash
Copy code
pip install -r requirements.txt
Usage
Run the script with the required arguments:

bash
Copy code
python main.py --mask_path mask_1920_1080.png --video_path data/parking_1920_1080.mp4 --model_path model/model.p
Arguments
--mask_path: Path to the mask image file (e.g., mask_1920_1080.png).
--video_path: Path to the parking video file (e.g., data/parking_1920_1080.mp4).
--model_path: Path to the pre-trained SVC model (e.g., model/model.p).
Output
The script processes the video feed and:

Detects parking spots defined in the mask file.
Displays a real-time video feed with:
Green Boxes: Available parking spots.
Red Boxes: Occupied parking spots.
Displays the count of available spots on the video.
Press q to exit the video display

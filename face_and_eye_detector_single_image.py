'''This script uses OpenCV's haarcascade (face and eye cascade) to detect face
and eyes in a given input image.'''

#Import necessary libraries
import cv2 
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
# Open video file using a different backend
cap = cv2.VideoCapture('C:/Users/NEM/drowsiness/face_and_eye_detector_single_image.py', cv2.CAP_FFMPEG)
# Rest of the code for video processing

# Define eye aspect ratio function
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio (EAR)
    ear = (A + B) / (2.0 * C)

    return ear

# Define drowsiness detection function
def detect_drowsiness(frame, detector, predictor, threshold):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the face in the frame using the Viola-Jones algorithm
    rects = detector(gray, 0)

    # Iterate over all the detected faces
    for rect in rects:
        # Predict the facial landmarks using the dlib shape predictor
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye landmarks
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Compute the eye aspect ratios
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Compute the average eye aspect ratio
        ear = (leftEAR + rightEAR) / 2.0

        # Check if the EAR is below the threshold
        if ear < threshold:
            return True

    # Return False if no drowsiness is detected
    return False

# Load the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define the left and right eye landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Define the drowsiness threshold
threshold = 0.25

# Start the video stream
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        break

    # Detect drowsiness in the frame
    drowsy = detect_drowsiness(frame, detector, predictor, threshold)

    # Display the result on the frame
    if drowsy:
        cv2.putText(frame, "DROWSY", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "AWAKE", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Frame", frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Quit if the 'q' key is pressed
    if key == ord("q"):
        break

# Release the video stream and close all windows
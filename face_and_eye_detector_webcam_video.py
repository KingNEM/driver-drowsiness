'''This script uses OpenCV's haarcascade (face and eye cascade) to detect face
and eyes in a video feed which can be inputted through a webcam.'''

#Import necessary libraries
import cv2 as cv
import numpy as np
import cv2
import dlib
from scipy.spatial import distance
import cv2

# Rest of the code for video processing


#Load face cascade and hair cascade from haarcascades folder
face_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
	@@ -39,7 +46,7 @@
#Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv.destroyAllWindows()

'''This script uses OpenCV's haarcascade (face and eye cascade) to detect face
and eyes in a video feed which can be inputted through a webcam.'''

	@@ -80,4 +87,4 @@
#Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv.destroyAllWindows()
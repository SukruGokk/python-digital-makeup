# @date: 23.03.2021
# @author: Şükrü Erdem Gök
# @version: Python 3.8
# @os: Windows 10
# @github: https://github.com/SukruGokk

# Make up

# Libs
import cv2
import dlib
import numpy as np
from sys import argv

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Get img
if argv[1] == 'cam':
	src = cv2.VideoCapture(0)
	_, frame = src.read()
else: 
	frame = imread(argv[1])


# Update
while True:

    _, frame = src.read()

    ff = frame.copy()

    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)

    for face in faces:
        # Create landmark object
        landmarks = predictor(image=gray, box=face)

        list = []

        # Loop through all the points

        iList = []

        for i in range(48, 68):iList.append(i)

        iList.reverse()

        for n in iList:
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            list.append([x, y])

        list.append([landmarks.part(59).x, landmarks.part(59).y])
        list.append([landmarks.part(60).x, landmarks.part(60).y])

        cv2.fillPoly(frame, [np.array(list, np.int32)], (93, 93, 207))

        alpha = 0.1

        frame = cv2.addWeighted(ff, alpha, frame, 1 - alpha, 0)

        for n in range(35, 39):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            if n == 35:
                x = landmarks.part(36).x-5
                y = landmarks.part(36).y-3

            cv2.line(frame, (x, y-3), (landmarks.part(n+1).x, landmarks.part(n+1).y-3), (0, 0, 0), 3)

        for j in range(42, 46):
            x = landmarks.part(j).x
            y = landmarks.part(j).y

            if j == 45:
                x = landmarks.part(45).x
                y = landmarks.part(45).y-3

                cv2.line(frame, (x, y - 3), (x+5, y-3), (0, 0, 0), 3)

            else:
                cv2.line(frame, (x, y - 3), (landmarks.part(j + 1).x, landmarks.part(j + 1).y - 3), (0, 0, 0), 3)

        alpha = 0.3

        frame = cv2.addWeighted(ff, alpha, frame, 1 - alpha, 0)

    # show the image
    cv2.imshow('Face', frame)

    # Exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        break

# When everything done, release the video capture and video write objects
cap.release()

# Close all windows
cv2.destroyAllWindows()
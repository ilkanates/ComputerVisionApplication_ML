import cv2
import os

# Load the Haar Cascade for object detection
cascade = cv2.CascadeClassifier('haarcascade_mcs_box_detector.xml')

# Load Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect objects in the image
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50), maxSize=(200, 200))

    # Draw boxes around the detected objects
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Show the image
    cv2.imshow('Image', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

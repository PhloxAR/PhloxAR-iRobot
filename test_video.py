import numpy as np
import cv2

# Capture video from camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Capture frame-by-frame
    _, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, thresh1 = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY_INV)
    _, thresh3 = cv2.threshold(frame, 127, 255, cv2.THRESH_TRUNC)
    _, thresh4 = cv2.threshold(frame, 127, 255, cv2.THRESH_TOZERO)
    _, thresh5 = cv2.threshold(frame, 127, 255, cv2.THRESH_TOZERO_INV)


    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO',
              'TOZERO_INV']
    images = [frame, thresh1, thresh2, thresh3, thresh4, thresh5]



    # Display the resulting frame
    #cv2.imshow('frame', gray)

    for t in range(len(titles)):
        cv2.imshow(titles[t], images[t])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()

cv2.destroyAllWindows()
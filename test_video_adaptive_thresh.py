import numpy as np
from datetime import datetime
import cv2

# Capture video from camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Capture frame-by-frame
    _, img = cap.read()
    empty = np.zeros(img.shape, np.uint8)
    empty[:, :, :] = img[:, :, :]

    #frame = cv2.medianBlur(frame, 3)
    gray = frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)

    _, th1 = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', 'Canny Edges']
    images = [img, th1, th2, th3, edges]

    res = cv2.bitwise_not(th1)

    _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(empty, contours, -1, (0, 255, 0), 2)

    #cv2.imshow('with edges', empty)

    cv2.imshow('NOT', res)

    for t in range(len(titles)):
        cv2.imshow(titles[t], images[t])

    cv2.imshow('contours', empty)

    key = cv2.waitKey(1) & 0xFF

    date = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f')

    if key == ord('q'):
        break
    elif key == ord('c'):
        cv2.imwrite(date + '_contour_.jpg', con_img)
        cv2.imwrite(date + '_ori_.jpg', img)
        cv2.imwrite(date + '_thresh_binary_not_.jpg', res)
        cv2.imwrite(date + '_gray_.jpg', frame)
        cv2.imwrite(date + '_thresh_binary_.jpg', th1)
        cv2.imwrite(date + '_thresh_mean_.jpg', th2)
        cv2.imwrite(date + '_thresh_gaussian_.jpg', th3)

# When everything done, release the capture
cap.release()

cv2.destroyAllWindows()


# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
#import argparse
import datetime
#import imutils
import time
import cv2
import numpy as np
from time import time, strftime


nFrames = 400

camera = cv2.VideoCapture("square.mkv")

cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)

avg_sum = np.zeros((1080,1920,3), np.float32)

for _ in range(nFrames):
        # grab the current frame and initialize the occupied/unoccupied
	(grabbed, frame) = camera.read()
        if frame is None:
                break
	alpha = 1.0
	beta = 1.0/nFrames
	avg_sum = cv2.addWeighted(avg_sum, alpha, frame.astype(np.float32), beta, 0.0)


camera.release()
#cv2.imshow('dst_rt', avg_sum.astype(np.uint8))
#cv2.waitKey(0)



camera = cv2.VideoCapture("square.mkv")

done_frame = avg_sum

for i in range(nFrames):
        # grab the current frame and initialize the occupied/unoccupied

	(grabbed, frame) = camera.read()
	if i % 7 != 0: #7 is good
		continue

        if frame is None:
                break

	diff = np.round(frame.astype(np.float32)-avg_sum)
	diff[diff < 0] = 0
#	diff = cv2.absdiff(frame.astype(np.float32), avg_sum)

	done_frame += diff

cv2.imwrite("background_with_drones.png", done_frame.astype(np.uint8))
cv2.imshow('dst_rt', done_frame.astype(np.uint8))
cv2.waitKey(0)

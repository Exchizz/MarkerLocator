from MarkerTracker import MarkerTracker
import numpy as np
import cv2
import cv

from time import sleep

kernelSize = 31
order=5

markerTracker = MarkerTracker(5, kernelSize, 1)

(temp_directionReal, temp_directionImag) = markerTracker.generateSymmetryDetectorKernel(order, kernelSize)
#(temp_directionReal, temp_directionImag) = markerTracker.generateSymmetryDetectorKernel(1, kernelSize)

temp_complex = np.array(temp_directionReal + 1j*temp_directionImag, dtype=complex)
absolute = np.absolute(temp_complex)
threshold = 0.4*absolute.max()


for i in range(0,314,10):
	print "i: ", float(i)/100.0
	phase = np.exp(float(i)/100*1j)


	t1_temp = temp_complex*np.power(phase, order)
	t1 = t1_temp.real > threshold
	
	t2_temp = temp_complex*np.power(phase, order)
	t2 = t2_temp.real < -threshold

	image = t1.astype(np.float32)-t2.astype(np.float32)

	image = image+1
	image = image*127
	print image
	vis = cv.fromarray(image)

	cv.SaveImage("Kernel_%d.jpg" % i, vis)
#cv.NamedWindow('temp_kernel', cv.CV_WINDOW_NORMAL)
#cv.ShowImage('temp_kernel', vis)


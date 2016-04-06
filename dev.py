from MarkerTracker import MarkerTracker
import numpy as np
import cv2
import cv

from time import sleep

kernelSize = 31
order=5

markerTracker = MarkerTracker(5, kernelSize, 1)

(kernel_order_n_Real, kernel_order_n_Imag) = markerTracker.generateSymmetryDetectorKernel(order, kernelSize)
(kernel_order_1_Real, kernel_order_1_Imag) = markerTracker.generateSymmetryDetectorKernel(1, kernelSize)

kernel_order_1_array = np.array(kernel_order_1_Real + 1j*kernel_order_1_Imag, dtype=complex)
kernel_order_n_array = np.array(kernel_order_n_Real + 1j*kernel_order_n_Imag, dtype=complex)

absolute = np.absolute(kernel_order_n_array)
threshold = 0.4*absolute.max()


phase = np.exp(0.1*3.14*1j)

t1_temp = kernel_order_n_array*np.power(phase, order)
t1 = t1_temp.real > threshold
	
t2_temp = kernel_order_n_array*np.power(phase, order)
t2 = t2_temp.real < -threshold

img_t1_t2_diff = t1.astype(np.float32)-t2.astype(np.float32)


angleThreshold = 3.14/(2*order)
print angleThreshold
# t3 = angle(tempDirection * phase) < angleThreshold;
# t4 = angle(tempDirection * phase) > -angleThreshold;
# mask = 1 - 2 * (t3 & t4);
# imagesc(mask);

t3 = np.angle(kernel_order_1_array * phase) < angleThreshold
t4 = np.angle(kernel_order_1_array * phase) > -angleThreshold

mask = 1-2*(t3 & t4)
#mask = (image+1)*127
image_without_arm = (img_t1_t2_diff) * mask


image = (img_t1_t2_diff+1)*127
t1_t2_diff = cv.fromarray(image)
cv.SaveImage("Kernel_%d.jpg" % 1, t1_t2_diff)


image_without_arm = (image_without_arm+1)*127
mask_img = cv.fromarray(image_without_arm)
cv.SaveImage("Kernel_%d.jpg" % 2, mask_img)


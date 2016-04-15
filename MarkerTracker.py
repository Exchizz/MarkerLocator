# -*- coding: utf-8 -*-
"""
Marker tracker for locating n-fold edges in images using convolution.

@author: Henrik Skov Midtiby
"""
import cv2.cv as cv
import numpy as np
import math
from time import sleep

class MarkerTracker:
    '''
    Purpose: Locate a certain marker in an image.
    '''
    def __init__(self, order, kernelSize, scaleFactor):
        self.kernelSize = kernelSize

        (kernelReal, kernelImag) = self.generateSymmetryDetectorKernel(order, kernelSize)
        self.order = order
        self.matReal = cv.CreateMat(kernelSize, kernelSize, cv.CV_32FC1)
        self.matImag = cv.CreateMat(kernelSize, kernelSize, cv.CV_32FC1)
        for i in range(kernelSize):
            for j in range(kernelSize):
                self.matReal[i, j] = kernelReal[i][j] / scaleFactor
                self.matImag[i, j] = kernelImag[i][j] / scaleFactor
        self.lastMarkerLocation = (None, None)
        self.orientation = None

        (kernelRealThirdHarmonics, kernelImagThirdHarmonics) = self.generateSymmetryDetectorKernel(3*order, kernelSize)
        self.matRealThirdHarmonics = cv.CreateMat(kernelSize, kernelSize, cv.CV_32FC1)
        self.matImagThirdHarmonics = cv.CreateMat(kernelSize, kernelSize, cv.CV_32FC1)
        for i in range(kernelSize):
            for j in range(kernelSize):
                self.matRealThirdHarmonics[i, j] = kernelRealThirdHarmonics[i][j] / scaleFactor
                self.matImagThirdHarmonics[i, j] = kernelImagThirdHarmonics[i][j] / scaleFactor

	# Create kernel used to remove arm in quality-measure
	(kernelRemoveArmReal, kernelRemoveArmImag) = self.generateSymmetryDetectorKernel(1, kernelSize)
	self.kernelComplex = np.array(kernelReal + 1j*kernelImag, dtype=complex)
	self.KernelRemoveArmComplex = np.array(kernelRemoveArmReal + 1j*kernelRemoveArmImag, dtype=complex)

	# Values used in quality-measure
	absolute = np.absolute(self.kernelComplex)
	self.threshold = 0.4*absolute.max()

        self.quality = 0
	self.y1 = int(math.floor(float(self.kernelSize)/2))
	self.y2 = int(math.ceil(float(self.kernelSize)/2))

	self.x1 = int(math.floor(float(self.kernelSize)/2))
	self.x2 = int(math.ceil(float(self.kernelSize)/2))

                  
    def generateSymmetryDetectorKernel(self, order, kernelsize):
        valueRange = np.linspace(-1, 1, kernelsize);
        temp1 = np.meshgrid(valueRange, valueRange)
        kernel = temp1[0] + 1j*temp1[1];
            
        magni = abs(kernel);
        kernel = kernel**order;
        kernel = kernel*np.exp(-8*magni**2);
         
        return (np.real(kernel), np.imag(kernel))

    def allocateSpaceGivenFirstFrame(self, frame):
        self.newFrameImage32F = cv.CreateImage((frame.width, frame.height), cv.IPL_DEPTH_32F, 3)
        self.frameReal = cv.CreateImage ((frame.width, frame.height), cv.IPL_DEPTH_32F, 1)
        self.frameImag = cv.CreateImage ((frame.width, frame.height), cv.IPL_DEPTH_32F, 1)
        self.frameRealThirdHarmonics = cv.CreateImage ((frame.width, frame.height), cv.IPL_DEPTH_32F, 1)
        self.frameImagThirdHarmonics = cv.CreateImage ((frame.width, frame.height), cv.IPL_DEPTH_32F, 1)
        self.frameRealSq = cv.CreateImage ((frame.width, frame.height), cv.IPL_DEPTH_32F, 1)
        self.frameImagSq = cv.CreateImage ((frame.width, frame.height), cv.IPL_DEPTH_32F, 1)
        self.frameSumSq = cv.CreateImage ((frame.width, frame.height), cv.IPL_DEPTH_32F, 1)


	self.quality_match = cv.CreateImage((1,1), cv.IPL_DEPTH_32F, 1)
    
    def locateMarker(self, frame):
        self.frameReal = cv.CloneImage(frame)
        self.frameImag = cv.CloneImage(frame)
        self.frameRealThirdHarmonics = cv.CloneImage(frame)
        self.frameImagThirdHarmonics = cv.CloneImage(frame)

        # Calculate convolution and determine response strength.
        cv.Filter2D(self.frameReal, self.frameReal, self.matReal) # src, dst, kernel
        cv.Filter2D(self.frameImag, self.frameImag, self.matImag) # src, dst, kernel

        cv.Mul(self.frameReal, self.frameReal, self.frameRealSq) # src, src, dst
        cv.Mul(self.frameImag, self.frameImag, self.frameImagSq) # src, src, dst
        cv.Add(self.frameRealSq, self.frameImagSq, self.frameSumSq)

        # Calculate convolution of third harmonics for quality estimation.
        cv.Filter2D(self.frameRealThirdHarmonics, self.frameRealThirdHarmonics, self.matRealThirdHarmonics)
        cv.Filter2D(self.frameImagThirdHarmonics, self.frameImagThirdHarmonics, self.matImagThirdHarmonics)
        
        min_val, max_val, min_loc, max_loc = cv.MinMaxLoc(self.frameSumSq)
        self.lastMarkerLocation = max_loc
        (xm, ym) = max_loc
        self.determineMarkerOrientation(frame)
	self.determineMarkerQuality_naive(frame)
#	self.determineMarkerQuality_Mathias(frame)
#        self.determineMarkerQuality()
        return max_loc

    def determineMarkerQuality_naive(self, frame_org):

	phase = np.exp((self.limitAngleToRange(-self.orientation))*1j)

	t1_temp = self.kernelComplex*np.power(phase, self.order)
	t1 = t1_temp.real > self.threshold

	t2_temp = self.kernelComplex*np.power(phase, self.order)
	t2 = t2_temp.real < -self.threshold

	img_t1_t2_diff = t1.astype(np.float32)-t2.astype(np.float32)

	angleThreshold = 3.14/(2*self.order)

	t3 = np.angle(self.KernelRemoveArmComplex * phase) < angleThreshold
	t4 = np.angle(self.KernelRemoveArmComplex * phase) > -angleThreshold
	mask = 1-2*(t3 & t4)

	template = (img_t1_t2_diff) * mask
	template = cv.fromarray(1-template)

	(xm, ym) = self.lastMarkerLocation


	y1 = ym - int(math.floor(float(self.kernelSize/2)))
	y2 = ym + int(math.ceil(float(self.kernelSize/2)))

	x1 = xm - int(math.floor(float(self.kernelSize/2)))
	x2 = xm + int(math.ceil(float(self.kernelSize/2)))



	try:
		frame = frame_org[y1:y2, x1:x2]
	except(TypeError):
		self.quality = 0
		return
	w,h = cv.GetSize(frame)
	im_dst = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)
	cv.Threshold(frame, im_dst, 128, 1, cv.CV_THRESH_BINARY)


	matches = 0
	blacks = 0
	w,h = cv.GetSize(im_dst)
	for x in xrange(w):
		for y in xrange(h):
			if cv.Get2D(im_dst, y, x)[0] == 0: # if pixel is black
				blacks+=1
				if cv.Get2D(im_dst, y, x)[0] ==  cv.Get2D(template, y, x)[0]:
					matches+=1
			else:
				continue


#	self.quality = float(matches)/(w*h)
	self.quality = float(matches)/blacks

	im_dst = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)
	cv.Threshold(frame, im_dst, 115, 255, cv.CV_THRESH_BINARY)

#	cv.ShowImage("small_image", im_dst)
#	cv.ShowImage("temp_kernel", template)

	


    def determineMarkerQuality_Mathias(self, frame):

	phase = np.exp((self.limitAngleToRange(-self.orientation))*1j)
	angleThreshold = 3.14/(2*self.order)

	t1 = (self.kernelComplex*np.power(phase, self.order)).real > self.threshold
	t2 = (self.kernelComplex*np.power(phase, self.order)).real < -self.threshold

	img_t1_t2_diff = t1.astype(np.float32)-t2.astype(np.float32)

	t3 = np.angle(self.KernelRemoveArmComplex * phase) < angleThreshold
	t4 = np.angle(self.KernelRemoveArmComplex * phase) > -angleThreshold

	mask = 1-1*(t3 & t4)

	template = (((img_t1_t2_diff * mask))*255).astype(np.uint8)

	(xm, ym) = self.lastMarkerLocation


	#print "ym: ", ym, " xm: ", xm, " y1: ", self.y1, " y2:", self.y2, ",x1: ", self.x1, " x2:", self.x2
	try:
		frame_tmp = np.array(frame[ym-self.y1:ym+self.y2, xm-self.x1:xm+self.x2])
	except(TypeError):
		print "error"
		self.quality = 0.0
		return
	frame_copy = frame_tmp.copy() # .copy() solves bug: http://www.shuangrimu.com/6/
	frame_img = cv.fromarray(255-frame_tmp.astype(np.uint8))

	frame_w, frame_h = cv.GetSize(frame_img)

	template_copy = template[0:frame_h, 0:frame_w].copy()
	template = cv.fromarray(template_copy)


#	cv.ShowImage("temp_kernel", template)
#	cv.ShowImage("small_image", frame_img)

	cv.MatchTemplate(frame_img, template, self.quality_match, cv.CV_TM_CCORR_NORMED) # cv.CV_TM_CCORR_NORMED shows best results
	self.quality = self.quality_match[0,0]

    def determineMarkerOrientation(self, frame):
        (xm, ym) = self.lastMarkerLocation
        realval = cv.Get2D(self.frameReal, ym, xm)[0]
        imagval = cv.Get2D(self.frameImag, ym, xm)[0]
        self.orientation = (math.atan2(-realval, imagval) - math.pi / 2) / self.order

        maxValue = 0
        maxOrient = 0
        searchDist = self.kernelSize / 3
        for k in range(self.order):
            orient = self.orientation + 2 * k * math.pi / self.order
            xm2 = int(xm + searchDist*math.cos(orient))
            ym2 = int(ym + searchDist*math.sin(orient))
            if(xm2 > 0 and ym2 > 0 and xm2 < frame.width and ym2 < frame.height):
                try:
                    intensity = cv.Get2D(frame, ym2, xm2)
                    if(intensity[0] > maxValue):
                        maxValue = intensity[0]
                        maxOrient = orient
                except:
                    print("determineMarkerOrientation: error: %d %d %d %d" % (ym2, xm2, frame.width, frame.height))
                    pass

        self.orientation = self.limitAngleToRange(maxOrient)

    def determineMarkerQuality(self):
        (xm, ym) = self.lastMarkerLocation
        realval = cv.Get2D(self.frameReal, ym, xm)[0]
        imagval = cv.Get2D(self.frameImag, ym, xm)[0]
        realvalThirdHarmonics = cv.Get2D(self.frameRealThirdHarmonics, ym, xm)[0]
        imagvalThirdHarmonics = cv.Get2D(self.frameImagThirdHarmonics, ym, xm)[0]
        argumentPredicted = 3*math.atan2(-realval, imagval)
        argumentThirdHarmonics = math.atan2(-realvalThirdHarmonics, imagvalThirdHarmonics)
        argumentPredicted = self.limitAngleToRange(argumentPredicted)
        argumentThirdHarmonics = self.limitAngleToRange(argumentThirdHarmonics)
        difference = self.limitAngleToRange(argumentPredicted - argumentThirdHarmonics)
        strength = math.sqrt(realval*realval + imagval*imagval)
        strengthThirdHarmonics = math.sqrt(realvalThirdHarmonics*realvalThirdHarmonics + imagvalThirdHarmonics*imagvalThirdHarmonics)
        #print("Arg predicted: %5.2f  Arg found: %5.2f  Difference: %5.2f" % (argumentPredicted, argumentThirdHarmonics, difference))        
        #print("angdifferenge: %5.2f  strengthRatio: %8.5f" % (difference, strengthThirdHarmonics / strength))
        # angdifference \in [-0.2; 0.2]
        # strengthRatio \in [0.03; 0.055]
        self.quality = math.exp(-math.pow(difference/0.3, 2))
        #self.printMarkerQuality(self.quality)
        
    def printMarkerQuality(self, quality):
        stars = ""        
        if(quality > 0.5):
            stars = "**"
        if(quality > 0.7):
            stars = "***"
        if(quality > 0.9):
            stars = "****"
        print("quality = %d): %5.2f %s" % (self.order, quality, stars))
        
    def limitAngleToRange(self, angle):
        while(angle < math.pi):
            angle += 2*math.pi
        while(angle > math.pi):
            angle -= 2*math.pi
        return angle

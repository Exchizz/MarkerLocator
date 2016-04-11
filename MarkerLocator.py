#!/usr/bin/env python
from time import time, strftime
import sys
import os

import numbers
sys.path.append('/opt/ros/hydro/lib/python2.7/dist-packages')
import cv
import cv2
import math
import numpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from ImageAnalyzer import ImageAnalyzer
from TrackerInWindowMode import TrackerInWindowMode
from PerspectiveTransform import PerspectiveCorrecter
from MarkerPose import MarkerPose

'''
2012-10-10
Script developed by Henrik Skov Midtiby (henrikmidtiby@gmail.com).
Provided for free but use at your own risk.

2013-02-13
Structural changes allows simultaneous tracking of several markers.
Frederik Hagelskjaer added code to publish marker locations to ROS.
'''

PublishToROS = False

if PublishToROS:
    import rospy
    from geometry_msgs.msg import Point

class CameraDriver:
    '''
    Purpose: capture images from a camera and delegate procesing of the
    images to a different class.
    '''
    def __init__(self, markerOrders = [7, 8], defaultKernelSize = 21, scalingParameter = 2500, cameraDevice=0):
        # Initialize camera driver.
        # Open output window.
#        cv.NamedWindow('filterdemo', cv.CV_WINDOW_AUTOSIZE)
        cv.NamedWindow('filterdemo', cv.CV_WINDOW_NORMAL)
#        cv.NamedWindow('filterdemo', cv.CV_WINDOW_AUTOSIZE)
	cv.NamedWindow('temp_kernel', cv.CV_WINDOW_NORMAL)
	cv.NamedWindow('small_image', cv.CV_WINDOW_NORMAL)

	# If an interface has been given to the constructor, it's a camdevice and we need to set some properties. If it's not, it means it's a file.
	if(isinstance(cameraDevice, numbers.Integral)):
	        self.setFocus()
	        # Select the camera where the images should be grabbed from.
	        self.camera = cv.CaptureFromCAM(cameraDevice)
	else:
		self.camera = cv.CaptureFromFile(cameraDevice)
        self.setResolution()

        # Storage for image processing.
        self.currentFrame = None
        self.processedFrame = None
        self.running = True
        # Storage for trackers.
        self.trackers = []
        self.windowedTrackers = []
        self.oldLocations = []
        # Initialize trackers.
        for markerOrder in markerOrders:
            temp = ImageAnalyzer(downscaleFactor=1)
            temp.addMarkerToTrack(markerOrder, defaultKernelSize, scalingParameter)
            self.trackers.append(temp)
            self.windowedTrackers.append(TrackerInWindowMode(markerOrder, defaultKernelSize))
            self.oldLocations.append(MarkerPose(None, None, None, None))
        self.cnt = 0
        self.defaultOrientation = 0

    def setFocus(self):
        # Disable autofocus
        os.system('v4l2-ctl -d '+str(cameraDevice)+' -c focus_auto=0')
        
        # Set focus to a specific value. High values for nearby objects and
        # low values for distant objects.
        os.system('v4l2-ctl -d ' + str(cameraDevice) + ' -c focus_absolute=0')

        # sharpness (int)    : min=0 max=255 step=1 default=128 value=128
        os.system('v4l2-ctl -d ' + str(cameraDevice) + ' -c sharpness=200')

    
    def setResolution(self):
#        cv.SetCaptureProperty(self.camera, cv.CV_CAP_PROP_FRAME_WIDTH, 1280)
#        cv.SetCaptureProperty(self.camera, cv.CV_CAP_PROP_FRAME_HEIGHT, 720)
        cv.SetCaptureProperty(self.camera, cv.CV_CAP_PROP_FRAME_WIDTH, 1920)
        cv.SetCaptureProperty(self.camera, cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
        #cv.SetCaptureProperty(self.camera, cv.CV_CAP_PROP_FRAME_WIDTH, 2304)
        #cv.SetCaptureProperty(self.camera, cv.CV_CAP_PROP_FRAME_HEIGHT, 1536)

    def getImage(self):
        # Get image from camera.
       	self.currentFrame = cv.QueryFrame(self.camera)

    def processFrame(self):
        # Locate all markers in image.
       for k in range(len(self.trackers)):
            if(self.oldLocations[k].x is None or self.oldLocations[k].quality < 0.4 and self.oldLocations[k].quality != 0 ):
                # Previous marker location is unknown, search in the entire image.
		print "Lost track of marker, searching entire image"
                self.processedFrame = self.trackers[k].analyzeImage(self.currentFrame)
                markerX = self.trackers[k].markerLocationsX[0]
                markerY = self.trackers[k].markerLocationsY[0]
                order = self.trackers[k].markerTrackers[0].order
		quality = self.trackers[k].markerTrackers[0].quality
                self.oldLocations[k] = MarkerPose(markerX, markerY, self.defaultOrientation, quality, order)
            else:
                # Search for marker around the old location.
                self.windowedTrackers[k].cropFrame(self.currentFrame, self.oldLocations[k].x, self.oldLocations[k].y)
                self.oldLocations[k] = self.windowedTrackers[k].locateMarker()
                self.windowedTrackers[k].showCroppedImage()


    def publishImageFrame(self, RP):
        im = numpy.asarray(self.currentFrame[:,:])
        RP.publishImage(im)

    def drawDetectedMarkers(self):
        for k in xrange(len(self.trackers)):
            xm = self.oldLocations[k].x
            ym = self.oldLocations[k].y
            orientation = self.oldLocations[k].theta
            cv.Circle(self.processedFrame, (xm, ym), 4, (55, 55, 255), 2)
            xm2 = int(xm + 50*math.cos(orientation))
            ym2 = int(ym + 50*math.sin(orientation))
            cv.Line(self.processedFrame, (xm, ym), (xm2, ym2), (255, 0, 0), 2)

    def showProcessedFrame(self):
        cv.ShowImage('filterdemo', self.processedFrame)

    def resetAllLocations(self):
        # Reset all markers locations, forcing a full search on the next iteration.
        for k in range(len(self.trackers)):
            self.oldLocations[k] = MarkerPose(None, None, None, None)

        
    def handleKeyboardEvents(self):
        # Listen for keyboard events and take relevant actions.
        key = cv.WaitKey(20) 
        # Discard higher order bit, http://permalink.gmane.org/gmane.comp.lib.opencv.devel/410
        key = key & 0xff
        if key == 27: # Esc
            self.running = False
        if key == 114: # R
            print("Resetting")
            self.resetAllLocations()
        if key == 115: # S
            # save image
            print("Saving image")
            filename = strftime("%Y-%m-%d %H-%M-%S")
            cv.SaveImage("output/%s.png" % filename, self.currentFrame)

    def returnPositions(self):
        # Return list of all marker locations.
        return self.oldLocations

class RosPublisher:
    def __init__(self, markers):
        # Instantiate ros publisher with information about the markers that
        # will be tracked.
        self.pub = []
        self.markers = markers
        self.bridge = CvBridge()
        for i in markers:
            self.pub.append( rospy.Publisher('positionPuplisher' + str(i), Point, queue_size = 10)  )

        self.imagePub = rospy.Publisher("imagePublisher", Image, queue_size=10)

        rospy.init_node('FrobitLocator')

    def publishMarkerLocations(self, locations):
        j = 0
        for i in self.markers:
	    print("x: %8.3f y: %8.3f angle: %8.3f quality: %8.3f order: %s" % (locations[j].x, locations[j].y, locations[j].theta, locations[j].quality, locations[j].order))
            #ros function
#            self.pub[j].publish(  Point( locations[j].x, locations[j].y, locations[j].theta )  )
            self.pub[j].publish(  Point( locations[j].x, locations[j].y, locations[j].quality )  )
            j = j + 1


    def publishImage(self, Image):
        try:
                self.imagePub.publish(self.bridge.cv2_to_imgmsg(Image, 'bgr8'))
        except CvBridgeError, e:
                print e
def main():

    t0 = time()
    t1 = time()
    t2 = time()

    print 'function vers1 takes %f' %(t1-t0)
    print 'function vers2 takes %f' %(t2-t1)
    
    toFind = [6]

    if PublishToROS:
        RP = RosPublisher(toFind)

    #cd = CameraDriver(toFind, defaultKernelSize = 25) # Best in robolab.
#    cd = CameraDriver(toFind, defaultKernelSize = 25, cameraDevice="drone_flight.mkv")
    cd = CameraDriver(toFind, defaultKernelSize = 25, cameraDevice="recording_flight_with_5_marker_afternoon.mkv")
    t0 = time()
     

    pointLocationsInImage = [[1328, 340], [874, 346], [856, 756], [1300, 762]]
    realCoordinates = [[0, 0], [300, 0], [300, 250], [0, 250]]
    perspectiveConverter = PerspectiveCorrecter(pointLocationsInImage, realCoordinates)
     
    while cd.running:
        (t1, t0) = (t0, time())
        #print "time for one iteration: %f" % (t0 - t1)
        cd.getImage()
        cd.processFrame()
        cd.drawDetectedMarkers()
        cd.showProcessedFrame()
        cd.handleKeyboardEvents()
        y = cd.returnPositions()
        if PublishToROS:
            RP.publishMarkerLocations(y)
            cd.publishImageFrame(RP)
        else:
            pass
            #print y
            for k in range(len(y)):
                try:
                    poseCorrected = perspectiveConverter.convertPose(y[k])
                    print("x: %8.3f y: %8.3f angle: %8.3f quality: %8.3f order: %s" % (poseCorrected.x, poseCorrected.y, poseCorrected.theta, poseCorrected.quality, poseCorrected.order))
                    #print("%3d %3d %8.3f" % (y[0][0], y[0][1], y[0][2]))
                except:
                    pass
    print("Stopping")


main()

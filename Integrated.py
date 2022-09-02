# 10 trials - 10 lidar data & 10 images taken

import pandas as pd
import pickle
import matplotlib
from matplotlib import pylab, mlab, pyplot
plt = pyplot
from IPython.core.pylabtools import figsize, getfigs
import shapely
from shapely.geometry import Polygon, Point
import cv2 as cv
from celluloid import Camera

from lidar360 import Point, LidarKit
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
import math


# change size of figure drawn
pylab.rcParams['figure.figsize'] = (6.0, 9.0)

# Lidar data
angle_maps = []
dist_maps = []

# context parameters
n = 100  # number of particles to sample per iteration
w, h = 10.7, 15.24  # width and height of room in meters
μu, σu = [1.2, 0, 0], [0.1, 0.1, 0.01] # motion model, motion input
# room = Polygon([[0, 0], [w, 0], [w, h], [0, h], [0, 0]]) # drawing the room

# Setup SimpleBlobDetector parameters.
params = cv.SimpleBlobDetector_Params()
# Change Threshold
params.minThreshold = 10
params.maxThreshold = 220
# Filter by Inertia Ratio
params.filterByInertia = True
params.minInertiaRatio = 0.5
# Filter by Area
params.filterByArea = True
params.minArea = 500
params.maxArea = 1000000000
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.3
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.6
# Filter by Color
params.filterByColor = True
params.blobColor = 255

lower = np.array([29, 40, 10])
upper = np.array([60, 255, 255])

# drawing the the environment of the room
def draw_env():
    ax.plot((0,w), (0,0), 'g')
    ax.plot((0,0), (h,0), 'g')
    ax.plot((w,w), (h,0), 'g')
    ax.plot((0,w), (h,h), 'g')
    ax.grid()

# distance from each particle to the boundary
def distance_to_map(m, px, py):
    return m.boundary.distance(Point((px,py)))
    
def process_lidar(data):
    random_sample = np.random.choice(n, size=n, replace=False)
    sample_angle, sample_dist = np.array([data[i] for i in random_sample]).T
    sample_angle = radians(sample_angle)

    sx = sample_dist*cos(sample_angle)
    sy = sample_dist*sin(sample_angle)

    return (sx, sy)

fig, ax = plt.subplots()
ax.set_xlim((0, w))
ax.set_ylim((0, h))
    
videoCapture = cv.VideoCapture(0)

    
for i in range(10):

    draw_env()
    
    ret, image = videoCapture.read()
    blur = cv2.GaussianBlur(image, (21,21), 0)
    hsvframe = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsvframe, lower, upper)
    mask = cv2.erode(mask, None, iterations=6)
    mask = cv2.dilate(mask, None, iterations=8)
        
    detector = cv.SimpleBlobDetector_create(params)
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if(area > 10000):
        x, y, w, h = cv.boundingRect(contour)
        box_width = w
        
    # Get the number of blobs found
    blobCount = len(keypoints)
    print(blobCount, "found")
    
    if box_width:
        p = box_width                    # perceived width, in pixels
        w = BALLOON_WIDTH                # approx. actual width, in meters (pre-computed)
        f = FOCAL_LENGTH * SCALE         # camera focal length, in pixels (pre-computed)
        d = f * w / p
        print("Distance = %.3fm" % d)

        

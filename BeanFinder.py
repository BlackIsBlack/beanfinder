##Bean finding project

from __future__ import division
import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import cos, sin

green = (0,255,0)
def show(image):
    plt.figure(figsize = (10,10))
    plt.imshow(image, interpolation = 'nearest')
    
def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img
    
def find_biggest_contour(image):
    image = image.copy()
    im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key = lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1,255,-1)
    return biggest_contour, mask

def circle_contour(image, contour):
    image_with_ellipse = image.copy()
    ellipse = cv2.fitEllipse(contour)
    cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.LINE_AA)
    return image_with_ellipse
                          
def find_beans(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    max_dimension = max(image.shape)
    scale = 700/max_dimension
    image = cv2.resize(image, None, fx=scale, fy=scale)

    image_blur = cv2.GaussianBlur(image, (7,7), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
    
    min_blue = np.array([10,130,120])
    max_blue = np.array([15,250,250])
    maskcol = cv2.inRange(image_blur_hsv, min_blue, max_blue)
    min_blue2 = np.array([73,128,140])
    max_blue2 = np.array([48,213,228])
    maskbri = cv2.inRange(image_blur_hsv, min_blue2, max_blue2)

    mask = maskcol + maskbri

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel)

    find_beans, mask_beans = find_biggest_contour(mask_clean)

    overlay = overlay_mask(mask_clean, image)

    circled = circle_contour(overlay, find_beans)
    show(circled)
    bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)
    return bgr
image = cv2.imread('beans.jpg')
result = find_beans(image)

cv2.imwrite('beans2.jpg',result)

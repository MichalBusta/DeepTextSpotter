'''
Created on Mar 1, 2017

@author: Michal.Busta at gmail.com
'''

import cv2
import numpy as np

import matplotlib.pyplot as plt

def draw_box_points(img, points, color = (0, 255, 0), thickness = 1):
  
  try:  
    cv2.line(img, (points[0][0], points[0][1]), (points[1][0], points[1][1]), color, thickness)
    cv2.line(img, (points[2][0], points[2][1]), (points[1][0], points[1][1]), color, thickness)
    cv2.line(img, (points[2][0], points[2][1]), (points[3][0], points[3][1]), color, thickness)
    cv2.line(img, (points[0][0], points[0][1]), (points[3][0], points[3][1]), color, thickness)
  except:
    pass
  
def draw_intersection_points(img, points, color = (0, 255, 0), thickness=1):
    
    for i in range(points.shape[0] - 1):
        cv2.line(img, (points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]), color, thickness)
    
    
    
def draw_detections(img, detections, color = (0, 255, 0)):
    
    for i in range(0, detections.shape[2]):
        det_word = detections[0, 0, i]
        if (det_word[0] == 0 and det_word[1] == 0) or det_word[5] < 0.05:
            break
        
        
        box  = ((det_word[0], det_word[1]), (det_word[2], det_word[3]), det_word[4] * 180 / 3.14)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        
        draw_box_points(img, box, color)
        
        
def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data, interpolation='nearest'); plt.axis('off')
    
    
def vis_im_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    
    # tile the filters into an image
    plt.imshow(data, interpolation='nearest'); plt.axis('off')
    
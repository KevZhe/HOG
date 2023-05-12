import argparse
import numpy as np
from PIL import Image
import os
import cv2


#parser for image input
parser = argparse.ArgumentParser(description='HOG')
parser.add_argument('--image', type=str, default='example.jpg', help='Path to image.')
args = parser.parse_args()

# 3x3 masks for sobel operator
Gx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
])

Gy = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
])

#converts rgb image into gray scale
def rgb_to_gray(img):
    
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

    grayscale = (0.299 * R + 
                 0.587 * G + 
                 0.114 * B).round(decimals=0)

    return grayscale

# Functions
def convolve(mat, filtermask):
    """
    Performs convolutions given an input and kernel
    Args:
        mat (numpy.ndarray): input array 
        filtermask (numpy.ndarray): kernel
    Returns:
        numpy.ndarray: result of convolutions
    """

    # get number of rows and cols in matrix
    rows, cols = mat.shape[0], mat.shape[1]

    # calculate the range of the mask
    radius, length = int(filtermask.shape[0] / 2), filtermask.shape[0]

    # initialize result
    res = np.empty(mat.shape)
    res[:] = 0

    # perform convolutions
    for i in range(rows - length + 1):
        for j in range(cols - length + 1):
            res[i+radius,j+radius] = np.sum(filtermask * mat[i:i+length, j:j+length])

    return res

#perform min-max normalization to make matrix values [0,255]
def min_max_normalization(mat):

    #get min and max val
    min_val = np.min(mat)
    max_val = np.max(mat)

    #scale matrix
    scaled = ((mat - min_val) /  (max_val-min_val)) * 255

    #round off
    scaled = scaled.round(decimals = 0)

    return scaled

#perform gradient operator on img
def gradient_operator(img):

    #convolve image with sobel operator
    grad_x, grad_y = convolve(img, Gx), convolve(img, Gy)

    #compute gradient magnitude
    grad_mag =  np.sqrt(np.square(grad_x) + np.square(grad_y))

    #normalize gradient magnitude to [0,255]
    grad_mag_normalized = min_max_normalization(grad_mag)

    #compute gradient angles
    gradient_angles = np.arctan(grad_y / grad_x) * (180/np.pi)

    #set values where both Gx and Gy are 0, to 0
    n, m =  img.shape[0], img.shape[1]
    for i in range(n):
        for j in range(m):
            if grad_x[i][j] == 0 and grad_y[i][j] == 0:
                grad_mag_normalized[i][j] = 0
                gradient_angles[i][j] = 0
    
    return grad_mag_normalized, gradient_angles

#histogram bins, bin # mapped to bin center
bins = {}
for i in range(1,10):
    bins[i] = 20*(i-1) + 10

#get histogram of oriented gradients from one cell (8x8)
def compute_histogram(cell):

    histogram = [0 for _ in range(10)]

    n =  cell.shape[0]

    for i in range(n):
        for j in range(n):
            angle =  cell[i,j]
            #stuff on the hw


def HOG(grad_mag_normalized, gradient_angles):
    #cell size = 8x8, block size = 16x16, step size  = 8 pixels

    #TODO 
    """
    loop over blocks, compute histograms for each cell in block, concat together
    block normalization (L2)
    leave histogram and final feature values as floating point numbers
    """

def main():
    #read in filename of image
    filename = args.image

    #read in image with RGB flag
    im = cv2.imread("data/" + filename, 1)
    
    print("Original Image Shape:", im.shape)

    #convert to gray scale
    gray = rgb_to_gray(im)
    
    print("Grayscale Image Shape:", gray.shape)

    grad_mag_normalized, gradient_angles = gradient_operator(gray)

if __name__ == "__main__":
    main()
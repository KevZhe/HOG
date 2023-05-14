import numpy as np
import cv2
import math

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


def rgb_to_gray(img):
    """
    Converts rgb image to gray scale
    Args:
        img (numpy.ndarray): input image

    Returns:
        numpy.ndarray: gray scale version of image
    """

    #extract red, green, and blue values from image
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

    #compute grayscale and round
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


def min_max_normalization(mat):
    """
    Performs min-max normalization to make matrix values [0,255]
    Args:
        mat (numpy.ndarray): input array 
    Returns:
        numpy.ndarray: normalized matrix
    """

    #get min and max val
    min_val = np.min(mat)
    max_val = np.max(mat)

    #scale matrix
    scaled = ((mat - min_val) /  (max_val-min_val)) * 255

    #round off
    scaled = scaled.round(decimals = 0)

    return scaled


def gradient_operator(img):
    """
    Perform gradient operator on img
    Args:
        img (numpy.ndarray): input image 
    Returns:
        grad_mag_normalized (numpy.ndarray): normalized magnitudes
        gradient_angles (numpy.ndarray): gradient angles
    """

    #convolve image with sobel operator Gx and Gy
    grad_x, grad_y = convolve(img, Gx), convolve(img, Gy)

    #compute gradient magnitude
    grad_mag =  np.sqrt(np.square(grad_x) + np.square(grad_y))

    #normalize gradient magnitude to [0,255]
    grad_mag_normalized = min_max_normalization(grad_mag)

    #compute gradient angles
    gradient_angles = np.arctan2(grad_y,grad_x) * (180/np.pi)

    #convert negative angles to positive
    gradient_angles = np.where(gradient_angles<0, gradient_angles + 360, gradient_angles)

    #set values where both grad_x and grad_x are 0, to 0
    n_rows, n_cols =  img.shape[0], img.shape[1]
    for i in range(n_rows):
        for j in range(n_cols):
            if grad_x[i,j] == 0 and grad_y[i,j] == 0:
                grad_mag_normalized[i,j] = 0
                gradient_angles[i,j] = 0

    #replace all nans with 0
    grad_mag_normalized = np.where(np.isnan(grad_mag_normalized), 0, grad_mag_normalized)
    gradient_angles = np.where(np.isnan(gradient_angles), 0, grad_mag_normalized)

    return grad_mag_normalized, gradient_angles


#histogram bins, bin # mapped to bin center
bin_centers = {}
for i in range(1,10):
    bin_centers[i] = 20*(i-1) + 10


def compute_histogram(cell_mag, cell_grad):
    """
    Computes histogram for a cell
    Args:
        cell_mag (numpy.ndarray): gradient magnitudes for cell  
        cell_grad (numpy.ndarray): gradient angles for cell  
    Returns:
        histogram (list): histogram weights for cell
    """

    #initialize histogram for cell
    histogram = [0 for _ in range(9)]

    n_rows, n_cols =  cell_mag.shape[0], cell_mag.shape[1]

    #calculate histogram for cell
    for i in range(n_rows):
        for j in range(n_cols):

            #get magnitude and angle
            magnitude, angle = cell_mag[i][j], cell_grad[i][j]

            #if angle >=180, subtract 180 from it
            if angle >= 180: 
                angle -= 180
            
            #angle is < 10 or angle > 170
            if angle < bin_centers[1] or angle >= bin_centers[9]:

                left_bin_idx = 9
                right_bin_idx = 1
            #otherwise find matching bins
            else:
                left_bin_idx = math.floor((angle - 10) / 20) + 1
                right_bin_idx = left_bin_idx + 1

            #if less than first bin center, use reference angle
            if angle < bin_centers[1]: angle += 180

            #print(magnitude, angle, left_bin_idx, right_bin_idx)

            #calculate adjustments
            histogram[left_bin_idx - 1] += magnitude * (1 - ((angle - bin_centers[left_bin_idx]) / 20))
            histogram[right_bin_idx - 1] += magnitude * ((angle - bin_centers[left_bin_idx]) / 20)

    #make sure that all numbers in histogram 
    assert np.all(np.array(histogram) >= 0)

    return histogram


def feature_of_block(block_magnitudes, block_gradient, cell_size):
    """
    Computes feature vector for block and performs L2 block normalization
    Args:
        block_magnitudes (numpy.ndarray): gradient magnitudes for block  
        block_gradient (numpy.ndarray): gradient angles for block  
        cell_size (int) : size of 1 side of cell
    Returns:
        block_histogram_normalized (numpy.array): normalized block feature vector
    """

    #initialize concatenation of all histograms of block
    block_histogram = np.empty((0,))

    #for all cells in block, compute histogram and add to 
    for i in range(0, block_magnitudes.shape[0], cell_size):
        for j in range(0, block_magnitudes.shape[1], cell_size):

            #get cell from block
            cell_mag, cell_grad  = block_magnitudes[i:i+cell_size, j:j+cell_size], block_gradient[i:i+cell_size, j:j+cell_size]

            #compute histogram of oriented gradients
            hist = np.array(compute_histogram(cell_mag, cell_grad))

            #concat to block histogram
            block_histogram = np.concatenate((block_histogram, hist))
    
    #L2 Normalization
    block_histogram_normalized = block_histogram / np.sqrt(np.sum(np.square(block_histogram)))

    #some blocks might have histograms with weight 0
    block_histogram_normalized = np.nan_to_num(block_histogram_normalized, nan = 0)

    return block_histogram_normalized


def HOG_feature_vector(grad_mag_normalized, gradient_angles, cell_size, block_size, step_size):
    """
    Computes HOG feature vector from all blocks
    Args:
        grad_mag_normalized (numpy.ndarray): gradient magnitudes for block  
        gradient_angles (numpy.ndarray): gradient angles for block  
        cell_size (int) : size of 1 side of cell
        block_size (int) : size of 1 side of block
        step_size (int) : block step size
    Returns:
        HOG_feature_normalized (numpy.array): normalized HOG feature vector
    """

    #cell size = 8x8, block size = 16x16, step size  = 8 pixels
    n_rows, n_cols = grad_mag_normalized.shape[0], grad_mag_normalized.shape[1]

    #final feature vector
    feature_vector = np.empty((0,))

    for i in range(0, n_rows - block_size + 1, step_size):
        for j in range(0, n_cols - block_size + 1, step_size):

            #fetch block magnitudes and gradients
            block_magnitudes, block_gradients = grad_mag_normalized[i:i+block_size, j:j+block_size], gradient_angles[i:i+block_size, j:j+block_size]
            #compute normalized block histogram
            block_histogram_normalized = feature_of_block(block_magnitudes, block_gradients, cell_size)
            #add histograms from block to feature vector
            feature_vector = np.concatenate((feature_vector, block_histogram_normalized))

    #normalized HOG Vector
    HOG_feature_normalized = feature_vector / np.sum(feature_vector)


    return HOG_feature_normalized, feature_vector


def HOG(filepath, normalized = True, magnitudes = False):
    """
    Computes HOG feature vector from image
    Args:
        filepath (string): filepath to image
    Returns:
        features (numpy.array): normalized HOG feature vector
    """

    #read in image with RGB flag
    im = cv2.imread(filepath, 1)

    #convert to gray scale
    gray = rgb_to_gray(im)

    #compute gradient magnitudes and gradient angles
    grad_mag_normalized, gradient_angles = gradient_operator(gray)

    #return normalized gradient magnitudes
    if magnitudes == True:
        return grad_mag_normalized

    #get final feature vector
    features_normalized, features = HOG_feature_vector(grad_mag_normalized, gradient_angles, cell_size = 8, block_size = 16, step_size = 8)

    #make sure there are no nan values in feature vector
    assert(np.count_nonzero(np.isnan(features_normalized)) == 0)

    #return normalized feature vector or non normalized version
    if normalized:
        return features_normalized
    else:
        return features
bin_centers = {}
for i in range(1,10):
    bin_centers[i] = 20*(i-1) + 10
import math

def func(mags, angles):
    histogram = [0 for _ in range(9)]
    for i in range(len(mags)):

            magnitude, angle = mags[i], angles[i]

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
            print("Reference Angle: %d, Left Index: %d, Right Index: %d " % (angle, left_bin_idx, right_bin_idx))
            #calculate 
            histogram[left_bin_idx - 1] += magnitude * (1 - ((angle - bin_centers[left_bin_idx]) / 20))
            histogram[right_bin_idx - 1] += magnitude * ((angle - bin_centers[left_bin_idx]) / 20)

    print("Final Histogram: \n", histogram)
func(mags = [10, 10], 
     angles = [348, 348])

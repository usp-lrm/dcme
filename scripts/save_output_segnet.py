#!/usr/bin/env python
import numpy as np
import argparse
import pickle
import shutil
import cv2
import sys
import os

caffe_root = './caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe



def compute_abs_value( distance_map ):
#    distance_map = np.transpose( distance_map, (1,2,0) )
#    print distance_map.shape
    x_vector = distance_map[0,:]
    y_vector = distance_map[1,:]
    magnitude_map = (x_vector**2 + y_vector**2)**.5
    max_value = np.max( magnitude_map )
    if max_value > 0:
        magnitude_map *= 255./max_value
    return magnitude_map


def save_distance_map( distance_map, prefix = '' ):
    file_name = prefix + 'distance_map'
    distance_file = open(file_name,'wb')
    pickle.dump(distance_map[0], distance_file)
    distance_file.close()
    return


# Group pixels > 0 if their distance is < max_distance
def cluster_centers_of_mass( vote_map, max_distance = 10 ):
    # Discard some pixels. Get top percentile
    pixels = np.where( vote_map > 0 )
#    threshold = np.percentile(vote_map[pixels], 95)
#    print '\nThreshold = {0}'.format( threshold )
#    print 'Max threshold = {0}'.format( max(vote_map[pixels]) )
    # Mean = 20
    threshold = 50

    rows, cols = np.where( vote_map > threshold )
    pixels = np.stack( (rows, cols), axis=1 )


    # Order pixels by row
    row_ind = np.argsort(pixels[:,0])
    # Compute differences
    diffs = np.diff(pixels[row_ind][:,0])
    # Group close numbers
    # Groups is a list of numpy arrays
    row_groups = np.split(pixels[row_ind], np.where(diffs > max_distance)[0] + 1)
    groups = []
    for row_group in row_groups:
        col_ind = np.argsort(row_group[:,1])
        diffs = np.diff(row_group[col_ind][:,1])
        col_groups = np.split(row_group[col_ind], np.where(diffs > max_distance)[0] + 1)
        for col_group in col_groups:
            groups.append( col_group )

    centers_of_mass = []
    for group in groups:
        center_of_mass = np.floor(np.mean(group, axis=0)).astype(int)
        centers_of_mass.append( center_of_mass )
    return centers_of_mass, threshold



def find_center_of_mass( distance_map ):
    # The center of mass is the center of the closed line with magnitude equals to level_threshold
    magnitude_map = compute_abs_value( distance_map )
    magnitude_map = np.array( magnitude_map, dtype=np.uint8 )
    level_threshold = 10
    deviation = 2
    pixels = np.where( magnitude_map > level_threshold + deviation )
    magnitude_map[pixels] = 0
    pixels = np.where( magnitude_map >= level_threshold - deviation )
    magnitude_map[pixels] = 255

    centers_of_mass = []
    ret, thresh = cv2.threshold(magnitude_map, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if hierarchy[0][i][2] != -1:
            cnt = contours[i]
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            centers_of_mass.append([int(y),int(x)])
    return centers_of_mass



def check_roi( x1, x2, y1, y2, num_rows, num_cols ):
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 >= num_cols:
        x2 = num_cols - 1
    if y2 >= num_rows:
        y2 = num_rows - 1
    return x1,x2,y1,y2



def extract_roi( image, mask, num_rows, num_cols ):
    rows, cols = np.where(mask > 0)
    x = np.min(cols)
    y = np.min(rows)
    w = np.max(cols) - x
    h = np.max(rows) - y
    large_side = max(w,h)
    # min roi size is 50x50
    if large_side <= 50:
        x1 = x + w/2 - 25
        y1 = y + h/2 - 25
        x2 = x1 + 50
        y2 = y1 + 50
        x1,x2,y1,y2 = check_roi( x1,x2,y1,y2, num_rows, num_cols )
        roi = image[y1:y2, x1:x2]
    else:
        x1 = x + w/2 - large_side/2
        y1 = y + h/2 - large_side/2
        x2 = x1 + large_side
        y2 = y1 + large_side
        x1,x2,y1,y2 = check_roi( x1,x2,y1,y2, num_rows, num_cols )
        roi = image[y1:y2, x1:x2]
    roi = cv2.resize( roi, (224,224), interpolation = cv2.INTER_AREA )
    return roi





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=False, default = 0)

    regression_model = './models/segnet_4_inference.prototxt'
    regression_weights = './weights/submission_3/segnet_4_iter_71000.caffemodel'   # 6.1  12.3 - third submission

#    color_list = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)] # , (128,128,128)] # -> grey
    # (255,  0,  0) -> red
    # (  0,255,  0) -> green
    # (  0,  0,255) -> blue
    # (255,255,  0) -> yellow
    # (255,  0,255) -> pink
    # (  0,255,255) -> cyan

    color_list = [
        (  0,  0,255), # blue
        (255,255,  0), # yellow
        (255,  0,255), # pink
        (255,  0,  0), # red
        (127,  0,255), # purple
        (  0,255,255), # cyan
        (  0,255,  0), # green
        (255,127,  0), # orange
        (  0,127,255), # blue 1
        (127,  0, 25), # marron
        (127,127,  0), # musgo
        (127,255,  0), # light green 2
        (255,127,127), # salmon
        (127,127,127), # grey
        (  0,127,127) # blue 2
#        (  0,  0,  0), # black
#        (255,255,255) # white
    ]

    dark_color_list = [(220,20,60),	(50,205,50), (0,0,128), (245,222,179), (0,206,209), (128,0,128) ,(112,128,144)]


    args = parser.parse_args()
    caffe.set_device( args.gpu )
    caffe.set_mode_gpu()

    net1 = caffe.Net(regression_model, regression_weights, caffe.TEST)


    text_file_name = args.test_file
    with open( text_file_name, 'r') as text_file:
        path_to_images = text_file.readlines()


    output_dir = './'
    results_dir = os.path.join(output_dir, 'results')
    if os.path.exists( results_dir ):
        shutil.rmtree( results_dir )
    os.makedirs( results_dir )

    magnitude_map_dir = os.path.join(output_dir, 'magnitude_map')
    if os.path.exists( magnitude_map_dir ):
        shutil.rmtree( magnitude_map_dir )
    os.makedirs( magnitude_map_dir )


    id_dictionary = {0:0, 1:24, 2:25, 3:26, 4:27, 5:28, 6:31, 7:32, 8:33 }

    counter = 1
    num_lines = len(path_to_images)
    for line in path_to_images:
        sys.stdout.write( "\rRunning inference (please wait): {0}/{1}".format(counter, num_lines) )
        sys.stdout.flush()
        counter += 1

        path_to_image = line[:-1]
        image_name = path_to_image.split('/')[-1]
        image_name = image_name[:-4]


        original_image = cv2.imread( path_to_image, cv2.IMREAD_COLOR )
        image = original_image/255.
        net1.blobs['data'].data[...] = np.transpose( image, (2,0,1) )
        regression_out = net1.forward()
        # output = conv1_1_D blob
        distance_map = regression_out['conv1_1_D_regress']
        # Expected distance_map shape = (1, 2, 340, 680)
        distance_map = distance_map[0]

        # Save magnitude map
        magnitude_map = compute_abs_value( distance_map )
        path_to_magnitude = os.path.join( magnitude_map_dir, image_name + '.png' )
        cv2.imwrite( path_to_magnitude, magnitude_map )

        distance_map = np.transpose( distance_map, (1,2,0) )
        round_map = np.array( distance_map, dtype=np.int8 )
        num_rows, num_cols = round_map.shape[:2]
        vote_map = np.zeros( (num_rows, num_cols) )
        rows, cols, _ = np.where( np.abs(round_map) > 0 )
        pixels = np.stack( (rows, cols), axis = 1)
        # Compute vote map, each pixel vote for one center of mass
        for pixel in pixels:
            row = pixel[0]
            col = pixel[1]
            x, y = round_map[row, col]
            cy = np.floor( row + y ).astype(int)
            cx = np.floor( col + x ).astype(int)
            if cy < num_rows and cx < num_cols:
                vote_map[cy, cx] += 1

        index = 0
#        # Each avg_centers_of_mass represents one instance
        avg_centers_of_mass, threshold = cluster_centers_of_mass( vote_map )
        for i in range(len(avg_centers_of_mass)):
            mask = np.zeros( (num_rows, num_cols), dtype=np.uint8 )
            for pixel in pixels:
                row = pixel[0]
                col = pixel[1]
                x, y = round_map[row, col]
                cy = np.floor( row + y ).astype(int)
                cx = np.floor( col + x ).astype(int)
                center_of_mass = [cy, cx]
                d1, d2 = np.array(avg_centers_of_mass[i]) - np.array(center_of_mass)
#                abs_dist = (d1**2 + d2**2)
#                if abs_dist < 100:
                abs_dist = (d1**2 + d2**2)**.5
                if abs_dist < 15:
                    mask[row,col] = 255

            # Get only external contours and fill internal holes
            ret, thresh = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask, contours, -1, 255, -1)

            pixels_mask = np.where( mask == 255 )
            
            original_image[pixels_mask] = color_list[index]
            index += 1
            if index >= len(color_list):
                index = 0

#            original_image = cv2.bitwise_and(original_image, original_image, mask=mask)
#            original_image = cv2.bitwise_or(original_image, original_image, mask=mask)
            
        path_to_output = os.path.join( results_dir, image_name + '.png' )
        cv2.imwrite( path_to_output, original_image )


print '\nSuccess!'


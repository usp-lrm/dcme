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
def cluster_centers_of_mass( vote_map, max_distance = 15 ):
    # Discard some pixels. Get top percentile
    pixels = np.where( vote_map > 0 )
#    threshold = np.percentile(vote_map[pixels], 95)
#    print '\nThreshold = {0}'.format( threshold )
#    print 'Max threshold = {0}'.format( max(vote_map[pixels]) )
    # Mean = 20
    vote_threshold = 30

    rows, cols = np.where( vote_map > vote_threshold )
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
    return centers_of_mass, vote_threshold



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

    # FCN-8
    regression_model = './models/fcn/fcn_inference.prototxt'
#    regression_weights = './weights/fcn_8_iter_34000.caffemodel' # 0.058 0.112
#    regression_weights = './weights/fcn_8_iter_26000.caffemodel' # 0.054 0.109
#    regression_weights = './weights/fcn_8_iter_40000.caffemodel' # 0.062 0.122
#    regression_weights = './weights/fcn_8_iter_42000.caffemodel' # 0.059
#    regression_weights = './weights/fcn_8_iter_46000.caffemodel' # 0.059
#    regression_weights = './weights/fcn_8_iter_50000.caffemodel' # 0.061 0.124
    regression_weights = './weights/fcn_8_iter_56000.caffemodel'


    classification_model = './models/classification/googlenet_deploy.prototxt'
    # fine imbalanced
#    classification_weights = './weights/classification/googlenet_iter_48000.caffemodel'
    # coarse balanced
    classification_weights = './weights/classification/googlenet_iter_56000.caffemodel'
    # fine balanced
#    classification_weights = './weights/classification/googlenet_iter_26000.caffemodel'

    args = parser.parse_args()
    caffe.set_device( args.gpu )
    caffe.set_mode_gpu()

    net1 = caffe.Net(regression_model, regression_weights, caffe.TEST)
    net2 = caffe.Net(classification_model, classification_weights, caffe.TEST)


    text_file_name = args.test_file
    with open( text_file_name, 'r') as text_file:
        path_to_images = text_file.readlines()


    output_dir = './'
#    output_dir = '/home/thomio/shared/datasets/cityscapes/detection/val'
#    output_dir = '/home/thomio/shared/datasets/cityscapes/detection/test'
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

        line = line[:-1]
        path_to_image, path_to_gt = line.split(' ')
        image_name = path_to_image.split('/')[-1]
        image_name = image_name[:-4]

        output_text_file_name = image_name + '.txt'
        path_to_output_text = os.path.join( results_dir, output_text_file_name )
        output_text_file = open(path_to_output_text, 'w')

        image = cv2.imread(path_to_image, cv2.IMREAD_COLOR )
        image = image/255.
        net1.blobs['data'].data[...] = np.transpose( image, (2,0,1) )
        regression_out = net1.forward()
        # output = conv1_1_D blob
        distance_map = regression_out['score']

        # Expected distance_map shape = (1, 2, 340, 680)
        distance_map = distance_map[0]

        # Save magnitude map
        magnitude_map = compute_abs_value( distance_map )
        path_to_magnitude = os.path.join( magnitude_map_dir, image_name + '.png' )
        cv2.imwrite( path_to_magnitude, magnitude_map )

#        avg_centers_of_mass = find_center_of_mass( distance_map )

#        distance_map = np.stack( (distance_map[0,:], distance_map[1,:]), axis=2 )
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

#        # Each avg_centers_of_mass represents one instance
        avg_centers_of_mass, threshold = cluster_centers_of_mass( vote_map )
        for i in range(len(avg_centers_of_mass)):
            output_mask = image_name + '_' + str(i) + '.png'
            path_to_output = os.path.join( results_dir, output_mask )
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

            resized_mask = cv2.resize( mask, (2048,1024), interpolation = cv2.INTER_CUBIC )
            cv2.imwrite( path_to_output, resized_mask )

            # Classify instance:
            # Check if mask exist
            if np.where(mask > 0)[0].size > 20:
                roi = extract_roi( image, mask, num_rows, num_cols )
                net2.blobs['data'].data[...] = np.transpose( roi, (2,0,1) )
                classification_out = net2.forward()
                probability_array = classification_out['prob']
                predicted_class = probability_array.argmax()
                probability = probability_array[0][ predicted_class ]

                msg = output_mask + ' ' + str(id_dictionary[predicted_class]) + ' '+ str(probability) + '\n'
                output_text_file.write( msg )

        output_text_file.close()
print '\nSuccess!'


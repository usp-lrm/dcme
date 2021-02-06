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
    distance_map = distance_map[0]
    x_vector = distance_map[0,:]
    y_vector = distance_map[1,:]
    absolute_map = (x_vector**2 + y_vector**2)**.5
    max_value = np.max( absolute_map )
    if max_value > 0:
        absolute_map *= 255./max_value
    return absolute_map


def save_distance_map( distance_map, prefix = '' ):
    file_name = prefix + 'distance_map'
    distance_file = open(file_name,'wb') 
    pickle.dump( distance_map[0] ,distance_file)   
    distance_file.close()
    return


# Group pixels >0 if their distance is < max_distance
def group_pixels( vote_map, max_distance = 5 ):
    # Discard some pixels. Get top percentile
    pixels = np.where( vote_map > 0 )
    theshold = np.percentile(vote_map[pixels], 95)

    rows, cols = np.where( vote_map > theshold )
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

    centroids = []
    for group in groups:
        centroid = np.floor(np.mean(group, axis=0)).astype(int)
        centroids.append( centroid )
    return centroids




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=False, default = 0)

    args = parser.parse_args()

    caffe.set_device( args.gpu )
    caffe.set_mode_gpu()
    net = caffe.Net( args.model, args.weights, caffe.TEST )


    text_file_name = args.test_file
    with open( text_file_name, 'r') as text_file:
        path_to_images = text_file.readlines()

#    output_text_file_name = 'classification_results.txt'
#    output_text_file = open(output_text_file_name, 'w')

    num_classes = 9
    num_labels = {}
    right_predictions = {}
    for i in range( num_classes ):
        label = str(i)
        num_labels[ label ] = 0
        right_predictions[ label ] = 0


    counter = 1
    num_lines = len(path_to_images)
    for line in path_to_images:
        sys.stdout.write( "\rRunning inference (please wait): {0}/{1}".format(counter, num_lines) )
        sys.stdout.flush()
        counter += 1


        line = line[:-1]
        path_to_image, class_gt = line.split(' ')
        image_name = path_to_image.split('/')[-1]
        image_name = image_name[:-4]

        image = cv2.imread( path_to_image, cv2.IMREAD_COLOR )
        image = image/255.
        net.blobs['data'].data[...] = np.transpose( image, (2,0,1) )
        out = net.forward()
        class_inference = out['prob'].argmax()

        num_labels[class_gt] += 1
        if class_gt == str(class_inference):
            right_predictions[class_gt] += 1


    print num_labels
    print right_predictions
    print '\nAccuracy per class:'
    for i in range( num_classes ):
        label = str(i)
        print 'Class {}, {:.2%} accuracy, {} labels'.format( i, right_predictions[label] / float(num_labels[label]), num_labels[label])
#   output_text_file.close()

print 'Success!'

#    {'1': 3395, '0': 21, '3': 4653, '2': 543, '5': 98, '4': 93, '7': 149, '6': 23, '8': 1165}
#    {'1': 1718, '0': 0, '3': 3817, '2': 1, '5': 0, '4': 1, '7': 0, '6': 0, '8': 705}

#    Accuracy per class:
#    Class 0, 21 labels, 0.00% accuracy
#    Class 1, 3395 labels, 50.60% accuracy
#    Class 2, 543 labels, 0.18% accuracy
#    Class 3, 4653 labels, 82.03% accuracy
#    Class 4, 93 labels, 1.08% accuracy
#    Class 5, 98 labels, 0.00% accuracy
#    Class 6, 23 labels, 0.00% accuracy
#    Class 7, 149 labels, 0.00% accuracy
#    Class 8, 1165 labels, 60.52% accuracy




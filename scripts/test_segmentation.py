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
    magnitude_map = (x_vector**2 + y_vector**2)**.5
    max_value = np.max( magnitude_map )
    if max_value > 0:
        magnitude_map *= 255./max_value
    return magnitude_map


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
    net = caffe.Net(args.model, args.weights, caffe.TEST)


    text_file_name = args.test_file
    with open( text_file_name, 'r') as text_file:
        path_to_images = text_file.readlines()


    output_dir = '/home/thomio/shared/datasets/cityscapes/detection/val/'

    results_dir = os.path.join(output_dir, 'results')
    if os.path.exists( results_dir ):
        shutil.rmtree( results_dir )
    os.makedirs( results_dir )

    id_dictionary = {0:0, 1:24, 2:25, 3:26, 4:27, 5:28, 6:31, 7:32, 8:33 }


    counter = 1
    num_lines = len(path_to_images)
    for line in path_to_images:
        sys.stdout.write( "\rRunning inference (please wait): {0}/{1}".format(counter, num_lines) )
        sys.stdout.flush()
#        print '{0}/{1}'.format(counter, num_lines)
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
        net.blobs['data'].data[...] = np.transpose( image, (2,0,1) )
        out = net.forward()
        # The output is the conv1_1_D blob
        distance_map = out['conv1_1_D_regress']
        class_map = out['class_prob']

        # Expected distance_map shape = (1, 2, 256, 512)
        distance_map = distance_map[0]
        # Expected class_map shape = (1, 9, 256, 512)
        class_map = class_map[0]


#        distance_map = np.stack( (distance_map[0,:], distance_map[1,:]), axis=2 )
        distance_map = np.transpose( distance_map, (1,2,0) )
        round_map = np.array( distance_map, dtype=np.int8 )
        vote_map = np.zeros( round_map.shape[:2] )
        rows, cols, _ = np.where( np.abs(round_map) > 0 )
        pixels = np.stack( (rows, cols), axis = 1)
        # Compute vote map, each pixel vote for one centroid
        for pixel in pixels:
            row = pixel[0]
            col = pixel[1]
            x, y = round_map[row, col]
            cy = np.floor( row + y ).astype(int)
            cx = np.floor( col + x ).astype(int)
            if cy < 512 and cx < 1024:
                centroid = [cy, cx]
                vote_map[cy, cx] += 1

        # Each avg_centroid represents one instance and class_map defines its class
        avg_centroids = group_pixels( vote_map )
        for i in range(len(avg_centroids)):
            output_mask = image_name + '_' + str(i) + '.png'
            path_to_output = os.path.join( results_dir, output_mask )

            mask = np.zeros( round_map.shape[:2], dtype=np.uint8 )
            for pixel in pixels:
                row = pixel[0]
                col = pixel[1]
                x, y = round_map[row, col]
                cy = np.floor( row + y ).astype(int)
                cx = np.floor( col + x ).astype(int)
                centroid = [cy, cx]
                d1, d2 = avg_centroids[i] - centroid
                abs_dist = (d1**2 + d2**2)**.5
                if abs_dist < 100:
                    mask[row,col] = 255

            # Each value from class_prediction represents a single class
            class_prediction = np.argmax(class_map, axis = 0)
            mask_pixels = np.where(mask > 0)
            classes, counts = np.unique( class_prediction[mask_pixels], return_counts = True )
            if classes.size != 0:
                most_voted_class = classes[ np.argmax(counts) ]
                # Class 0 is the background -> no need to generate mask for this
                if most_voted_class > 0:
                    most_voted_class_probabilities = class_map[most_voted_class]
                    probability_mean = np.mean(most_voted_class_probabilities[mask_pixels])

                    # Get only external contours to fill internal holes
                    ret, thresh = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
                    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(mask, contours, -1, 255, -1)
                    resized_mask = cv2.resize( mask, (2048,1024), interpolation = cv2.INTER_CUBIC )
                    cv2.imwrite( path_to_output, resized_mask )

                    msg = output_mask + ' ' + str(id_dictionary[most_voted_class]) + ' '+ str(probability_mean) + '\n'
                    output_text_file.write( msg )

        output_text_file.close()

print '\nSuccess!'




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
    parser.add_argument('--test_file', type=str, required=True)

    args = parser.parse_args()

    text_file_name = args.test_file
    with open( text_file_name, 'r') as text_file:
        path_to_images = text_file.readlines()


    output_dir = '/home/thomio/shared/datasets/cityscapes/detection/val/results'
    if os.path.exists( output_dir ):
        shutil.rmtree( output_dir )
    os.makedirs( output_dir )

    division_factor = 1000
    instance_counter = {}

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
        path_to_output_text = os.path.join( output_dir, output_text_file_name )
        output_text_file = open(path_to_output_text, 'w')

        gt_image = cv2.imread( path_to_gt, cv2.IMREAD_ANYDEPTH )
        instances = np.unique( gt_image )

        for i in range(len(instances)):
            instance_id = instances[i]
            output_mask = image_name + '_' + str(i) + '.png'
            path_to_output = os.path.join( output_dir, output_mask )

            # mask = np.zeros( (256,512), np.uint8 );
            mask = np.zeros( (340,680), np.uint8 );
            # mask = np.zeros( (512,1024), np.uint8 );
            if instance_id > division_factor:
                pixels = np.where( gt_image == instance_id )
                mask[pixels] = 255
                resized_mask = cv2.resize( mask, (2048,1024), interpolation = cv2.INTER_CUBIC )
                cv2.imwrite( path_to_output, resized_mask )

                mask_pixels = np.where(mask > 0)
                class_id, instance = divmod( instance_id, division_factor )
                msg = output_mask + ' ' + str(class_id) + ' 1\n'
                output_text_file.write( msg )

                if class_id in instance_counter:
                    instance_counter[class_id] += 1
                else:
                    instance_counter[class_id] = 1

        output_text_file.close()
    print '\nSuccess!'
    print instance_counter




#!/usr/bin/env python
import numpy as np
import argparse
import json
import cv2
import os



def evaluate_distribution( path_to_images ):
    class_dictionary = { 24:0, 25:1, 26:2, 27:3, 28:4, 31:5, 32:6, 33:7}
    all_classes = { 'person':0, 'rider':1, 'car':2, 'truck':3, 'bus':4, 'train':5, 'motorcycle':6, 'bicycle':7 }
    inv_classes = {v: k for k, v in all_classes.iteritems()}

    division_factor = 1000
    instance_counter = np.zeros( len(class_dictionary), dtype=np.uint32 )

    print 'Number of images = {}'.format( len(path_to_images) )
    for line in path_to_images:
        path_to_image, path_to_gt = line[:-1].split(' ')

        gt_image = cv2.imread( path_to_gt, cv2.IMREAD_ANYDEPTH )
        instances = np.unique( gt_image )
        for instance_id in instances:
            if instance_id > division_factor:
                class_id, instance = divmod( instance_id, division_factor )
                if class_id in class_dictionary.keys():
                    index = class_dictionary[class_id]
                    instance_counter[index] += 1

    num_instances = np.sum(instance_counter)
    print 'Number of instances = {}'.format( num_instances )
    distribution = np.divide(instance_counter, num_instances, dtype=np.float32)
    print 'Class distribution:'
    for i in range(len(distribution)):
        print '\t{0} = {1:.2%} ({2})'.format( inv_classes[i], distribution[i], instance_counter[i] )
    return




def balance_dataset( path_to_images ):
    class_dictionary = { 24:0, 25:1, 26:2, 27:3, 28:4, 31:5, 32:6, 33:7}
    large_classes = { 24:0, 25:1, 26:2, 33:7 }
    small_classes = { 27:3, 28:4, 31:5, 32:6 }
    division_factor = 1000
    # large_classs = 0
    # small_classes = 1
    file_dict = {0:[], 1:[]}

    balanced_dataset = []
    # Separate classes
    for line in path_to_images:
        path_to_image, path_to_gt = line[:-1].split(' ')

        gt_image = cv2.imread( path_to_gt, cv2.IMREAD_ANYDEPTH )
        instances = np.unique( gt_image )
        class_id_per_image = []
        for instance_id in instances:
            if instance_id > division_factor:
                class_id, instance = divmod( instance_id, division_factor )
                class_id_per_image.append(class_id)

        # Check if there is at leas one instance from a small class
        if any(map(lambda each: each in small_classes.keys(), class_id_per_image)):
            file_dict[1].append( line )
        else:
            file_dict[0].append( line )

    # Reduce large classes instances
    reduced_list = []
    for i in range(len(file_dict[0])):
        if i % 30 == 0:
            reduced_list.append( file_dict[0][i] )
    file_dict[0] = reduced_list

    # Increase small classes instances
    balanced_dataset += file_dict[0]
    balanced_dataset += file_dict[1]
    balanced_dataset += file_dict[1]
    balanced_dataset += file_dict[1]
    return balanced_dataset


if __name__ == '__main__':
    parameters_file_name = 'parameters.json'
    with open(parameters_file_name, 'r') as json_file:    
        parameters_data = json.load(json_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file', type=str, required=True)
    args = parser.parse_args()
    path_to_file = args.text_file

    print 'Text file = {}'.format( path_to_file )
    with open( path_to_file, 'r') as text_file:
        path_to_images = text_file.readlines()


    evaluate_distribution( path_to_images )
    balanced_dataset = balance_dataset( path_to_images )
    evaluate_distribution( balanced_dataset )

    output_txt_file = path_to_file[:-4] + '_balanced.txt'
    with open(output_txt_file, 'w') as file_handler:
        for line in balanced_dataset:
            file_handler.write( line )


#    Text file = dataset/cityscapes_train.txt
#    Number of images = 2975
#    Number of instances = 51933
#    Class distribution:
#	    person = 34.43% (17881)
#	    rider = 3.38% (1753)
#	    car = 51.76% (26881)
#	    truck = 0.93% (482)
#	    bus = 0.73% (379)
#	    train = 0.32% (168)
#	    motorcycle = 1.42% (735)
#	    bicycle = 7.04% (3654)

#    Text file = dataset/cityscapes_train_balanced.txt
#    Number of images = 3348
#    Number of instances = 68373
#    Class distribution:
#	    person = 36.91% (25235)
#	    rider = 3.73% (2553)
#	    car = 44.73% (30586)
#	    truck = 2.11% (1446)
#	    bus = 1.66% (1137)
#	    train = 0.74% (504)
#	    motorcycle = 3.22% (2205)
#	    bicycle = 6.88% (4707)




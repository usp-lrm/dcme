#!/usr/bin/env python
import numpy as np
import shutil
import json
import cv2
import os



def create_new_dir( new_directory ):
    if os.path.exists( new_directory ):
        shutil.rmtree( new_directory )
    os.makedirs( new_directory )
    return


def resize_image( path_to_image, path_to_resized_image ):
    image = cv2.imread( path_to_image )
#    image = cv2.resize( image, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA )
    image = cv2.resize( image, (680,340), interpolation = cv2.INTER_AREA )
    cv2.imwrite( path_to_resized_image, image )
    return


## For .json files:
#def resize_gt_from_json( path_to_polygon, path_to_resized_gt ):
#    with open( path_to_polygon ) as data_file:
#        data = json.load(data_file)
#    #print 'imgHeight = ', data['imgHeight']
#    #print 'imgWidth = ', data['imgWidth']

#    img_objects = data['objects']
#    cars_contours = []
#    for obj in img_objects:
#        if obj['label'] == 'car':
#            contour = obj['polygon']
#            cars_contours.append( contour )

#    ground_truth = np.zeros( (512,1024), np.uint8 );
#    for i in range(len(cars_contours)):
#        temp_ground_truth = np.zeros( (1024,2048), np.uint8 );
#        cnt = np.asarray(cars_contours[i])
#        # This is a wierd format for cv2 contours.
#        # (n,1,2) -> n = number of points
#        cnt = cnt.reshape(-1,1,2)
#        cv2.drawContours(temp_ground_truth, [cnt], 0, 1, -1)
#        # temp_ground_truth = cv2.resize( temp_ground_truth, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
#        temp_ground_truth = cv2.resize( temp_ground_truth, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
#        index = np.where( temp_ground_truth > 0 )
#        ground_truth[index] = i+1

#    cv2.imwrite( path_to_resized_gt, ground_truth )
#    return


def resize_gt( path_to_gt, path_to_resized_gt ):
    # This value was defined by cityscapes-dataset
    division_factor = 1000
    gt_image = cv2.imread( path_to_gt, cv2.IMREAD_ANYDEPTH )
    instances = np.unique( gt_image )

    ground_truth = np.zeros( (340,680), np.uint16 );
    for instance_id in instances:
        temp_ground_truth = np.zeros( gt_image.shape, np.uint16 );
        if instance_id > division_factor:
            pixels = np.where( gt_image == instance_id )
            temp_ground_truth[pixels] = 255
#            temp_ground_truth = cv2.resize( temp_ground_truth, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
            temp_ground_truth = cv2.resize( temp_ground_truth, (680,340), interpolation = cv2.INTER_AREA)
            pixels = np.where( temp_ground_truth > 0 )
            ground_truth[pixels] = instance_id

    cv2.imwrite( path_to_resized_gt, ground_truth )
    return


def resize_coarse_dataset( parameters_data, subdir ):
    contour_type = 'gtCoarse_instanceIds'
    camera = parameters_data['camera']

    path_to_dataset = parameters_data['path_to_dataset']
    path_to_images = os.path.join( path_to_dataset, 'leftImg8bit' )
    path_to_gt = os.path.join( path_to_dataset, 'gtCoarse' )

    output_directory = os.path.join(path_to_dataset, 'detection')


    text_directory = 'dataset'
    if not os.path.exists( text_directory ):
        os.makedirs( text_directory )
    text_file_name = 'cityscapes_coarse' + subdir + '.txt'
    path_to_text_file = os.path.join( text_directory, text_file_name )
    text_file = open( path_to_text_file, 'w' )

    cities_dir = os.path.join( path_to_images, subdir )
    output_images_dir = os.path.join(output_directory, subdir, 'resized_images')
    create_new_dir( output_images_dir )

    gt_cities_dir = os.path.join( path_to_gt, subdir )
    output_gt_dir = os.path.join(output_directory, subdir, 'ground_truth')
    create_new_dir( output_gt_dir )

    all_cities = os.listdir( cities_dir )
    counter = 1
    for city in all_cities:
        city_directory = os.path.join(cities_dir, city)
        all_files = os.listdir( city_directory )
        print counter, '/', len(all_cities), city_directory
        counter += 1
        for file_name in all_files:
            path_to_image = os.path.join( city_directory, file_name )
            path_to_resized_image = os.path.join( output_images_dir, file_name )
            resize_image( path_to_image, path_to_resized_image )

            gt_name = file_name.replace( camera, contour_type )
            path_to_gt = os.path.join( gt_cities_dir, city, gt_name )
            path_to_resized_gt =  os.path.join( output_gt_dir, gt_name )
            resize_gt( path_to_gt, path_to_resized_gt )

            msg = path_to_resized_image + ' ' + path_to_resized_gt + '\n'
            text_file.write( msg )
    text_file.close()
    return


if __name__ == '__main__':
    parameters_file_name = 'parameters.json'
    with open(parameters_file_name, 'r') as json_file:    
        parameters_data = json.load(json_file)

    print 'Resizing dataset, please wait'
    resize_coarse_dataset(parameters_data, 'train_extra')




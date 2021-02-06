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


def check_roi( x1, x2, y1, y2 ):
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > 2047:
        x2 = 2047
    if y2 > 1023:
        y2 = 1023
    return x1,x2,y1,y2



def extract_dataset( parameters_data, subdir, dataset_type = 'gtFine' ):

    contour_type = 'gtFine_instanceIds'

    if dataset_type == 'gtCoarse':
        contour_type = 'gtCoarse_instanceIds'
        subdir = 'train_extra'


    camera = parameters_data['camera']

    path_to_dataset = parameters_data['path_to_dataset']

    path_to_images = os.path.join( path_to_dataset, 'leftImg8bit' )
    path_to_gt = os.path.join( path_to_dataset, dataset_type )
 
    output_directory = os.path.join(path_to_dataset, 'classification', dataset_type )

    id_dictionary = { 24:1, 25:2, 26:3, 27:4, 28:5, 29:0, 30:0, 31:6, 32:7, 33:8}
    object_counter ={}
    for i in range( 9 ):
        object_counter[i] = 0


    text_directory = 'dataset'
    if not os.path.exists( text_directory ):
        os.makedirs( text_directory )
    text_file_name = 'cityscapes_' + dataset_type + '_class_' + subdir + '.txt'
    path_to_text_file = os.path.join( text_directory, text_file_name )
    text_file = open( path_to_text_file, 'w' )

    output_images_dir = os.path.join(output_directory, subdir)
    create_new_dir( output_images_dir )
    cities_dir = os.path.join( path_to_images, subdir )
    gt_cities_dir = os.path.join( path_to_gt, subdir )

    all_cities = os.listdir( cities_dir )
    counter = 1
    for city in all_cities:
        city_directory = os.path.join(cities_dir, city)
        all_files = os.listdir( city_directory )
        print counter, '/', len(all_cities), city_directory
        counter += 1
        for file_name in all_files:
            path_to_image = os.path.join( city_directory, file_name )
            image = cv2.imread( path_to_image )

            gt_name = file_name.replace( camera, contour_type )
            path_to_gt = os.path.join( gt_cities_dir, city, gt_name )
            # This value was defined by cityscapes-dataset
            division_factor = 1000
            gt_image = cv2.imread( path_to_gt, cv2.IMREAD_ANYDEPTH )
            instances = np.unique( gt_image )
            for instance_id in instances:
                if instance_id > division_factor:
                    class_id, instance = divmod( instance_id, division_factor )
                    pixels = np.where( gt_image == instance_id )

#                    image_mask = np.zeros( gt_image.shape, np.uint8 );
#                    image_mask[pixels] = 255
#                    temp_image = cv2.bitwise_and(image, image, mask = image_mask)

                    rows, cols = pixels
                    x = np.min(cols)
                    y = np.min(rows)
                    w = np.max(cols) - x
                    h = np.max(rows) - y
                    large_side = max(w,h)
                    if large_side <= 50:
                        extra_pixels = 0
                        # if rider
                        if class_id == 25:
                            extra_pixels = 10
                        x1 = x + w/2 - 25 - extra_pixels
                        y1 = y + h/2 - 25
                        x2 = x1 + 50 + extra_pixels
                        y2 = y1 + 50 + extra_pixels
                        x1,x2,y1,y2 = check_roi( x1,x2,y1,y2 )
                        roi = image[y1:y2, x1:x2]
                        roi = cv2.resize( roi, (224,224), interpolation = cv2.INTER_AREA)
                    else:
                        extra_pixels = 0
                        # if rider
                        if class_id == 25:
                            extra_pixels = 20
                        x1 = x + w/2 - large_side/2 - extra_pixels
                        y1 = y + h/2 - large_side/2
                        x2 = x1 + large_side + extra_pixels
                        y2 = y1 + large_side + extra_pixels
                        x1,x2,y1,y2 = check_roi( x1,x2,y1,y2 )
                        roi = image[y1:y2, x1:x2]
                        roi = cv2.resize( roi, (224,224), interpolation = cv2.INTER_AREA)

                    object_class = id_dictionary[class_id]
                    object_counter[object_class] += 1

                    save_instance = False
                    if object_class == 1:
                        if (object_counter[object_class] % 3) == 0:
                            save_instance = True
                    elif object_class == 3:
                        if (object_counter[object_class] % 6) == 0:
                            save_instance = True
                    elif object_class == 8:
                        if (object_counter[object_class] % 2) == 0:
                            save_instance = True
                    else:
                        save_instance = True


                    if save_instance:
                        new_image_name = file_name[:-4] + '_' + str(class_id) + '_' + str(instance) + '.png'
                        path_to_output_image = os.path.join( output_images_dir, new_image_name )
                        cv2.imwrite( path_to_output_image, roi )

                        msg = path_to_output_image + ' ' + str(object_class) + '\n'
                        text_file.write( msg )
    text_file.close()
    return


if __name__ == '__main__':
    parameters_file_name = 'parameters.json'
    with open(parameters_file_name, 'r') as json_file:    
        parameters_data = json.load(json_file)

    print 'Resizing dataset, please wait'
    dataset_type = 'gtFine'
    extract_dataset(parameters_data, 'val', dataset_type)
    extract_dataset(parameters_data, 'train', dataset_type)

    # Only use gtCoarse for the training dataset
#    dataset_type = 'gtCoarse'
#    extract_dataset(parameters_data, 'train', dataset_type)



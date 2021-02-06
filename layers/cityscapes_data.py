import numpy as np
import random
import cv2
import sys

caffe_root = './caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe


class CityscapesDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # Check top shape
        if len(top) != 3:
            raise Exception("Need to define top blobs (data and label)")
        
        #Check bottom shape
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
        
        #Read parameters
        params = eval(self.param_str)
        src_file_name = params["source"]
#        shuffle = params["shuffle"]
        self.batch_size = params["batch_size"]

        with open(src_file_name) as src_file:
            self.dataset = src_file.readlines()

#        if shuffle:
        random.shuffle(self.dataset)

        line = self.dataset[0]
        # Remove new line char
        line = line[:-1]
        path_to_image, path_to_gt = line.split(' ')
        image = cv2.imread(path_to_image, cv2.IMREAD_COLOR )
        top[0].reshape(self.batch_size, 3, image.shape[0], image.shape[1])
        top[1].reshape(self.batch_size, 2, image.shape[0], image.shape[1])
        top[2].reshape(self.batch_size, 1, image.shape[0], image.shape[1])

        self.index = 0
        return

        
    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            input_image, label, label2 = self.load_new_pair()
            top[2].data[itt, ...] = label2.reshape(1, label.shape[0], label.shape[1])
            top[1].data[itt, ...] = np.transpose( label, (2,0,1) )
            top[0].data[itt, ...] = np.transpose( input_image, (2,0,1) )
        return


    def load_new_pair(self):
        line = self.dataset[self.index]
        # Remove new line char
        line = line[:-1]
        path_to_image, path_to_gt = line.split(' ')
        self.index += 1
        if self.index >= len(self.dataset):
            self.index = 0

        image = cv2.imread(path_to_image, cv2.IMREAD_COLOR )
        image = image/255.

        gt_instances = cv2.imread( path_to_gt, cv2.IMREAD_ANYDEPTH )
        gt_distances, class_id = self.compute_distance_map( gt_instances )
        return image, gt_distances, class_id


    def compute_distance_map( self, gt_instances ):
        id_dictionary = { 24:1, 25:2, 26:3, 27:4, 28:5, 31:6, 32:7, 33:8}
        # This value was defined by cityscapes-dataset
        division_factor = 1000
        instances = np.unique( gt_instances )
        # 0 represents the background, it is not an instance
        instances = np.delete( instances, 0 )
        x_vector = np.zeros( gt_instances.shape )
        y_vector = np.zeros( gt_instances.shape )
        z_vector = np.zeros( gt_instances.shape )
        for instance in instances:
            pixels = np.where(gt_instances == instance)
            rows, cols = pixels
            x = np.min(cols)
            y = np.min(rows)
            w = np.max(cols) - x
            h = np.max(rows) - y
            area = w*h
            # Detections where the area is below a threshold are not accounted
            # But you can delete this condition... I think... maybe... hope so
            if area > 50:
                # centroid
                cx = x + w/2.
                cy = y + h/2.
                class_id, instance = divmod( instance, division_factor )
                # class_id 29 and 30 are not evaluated
                if class_id in id_dictionary:
                    new_id = id_dictionary[class_id]
                    z_vector[pixels] = new_id
                    for i in range( len(pixels[0]) ):
                        row = rows[i]
                        col = cols[i]
                        distance_x = (cx - col)
                        distance_y = (cy - row)
                        x_vector[row,col] = distance_x
                        y_vector[row,col] = distance_y

        gt_distances = np.stack( (x_vector, y_vector), axis=2)
        return gt_distances, z_vector


    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (img shape and batch size)
        
        If we were processing a fixed-sized number of images (for example in Testing)
        and their number wasn't a  multiple of the batch size, we would need to
        reshape the top blob for a smaller batch.
        """
        pass


    def backward(self, bottom, top):
        """
        This layer does not back propagate
        """
        pass

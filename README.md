### OVERVIEW
Source code for [DCME for instance segmentation](https://arxiv.org/abs/1711.09060).

Pease check [ISISA](https://arxiv.org/abs/1902.05498) for an improved version.


### TRAIN
 - Download and extract cityscapes
 - Download, compile and install caffe
 - Set the variables in parameters.json
 - set PYTHONPATH to point to caffe_layer

    ```
      source set_python_env.sh
    ```

 - Resize cityscapes images/annotations

    ```
      python scripts/resize_dataset.py
    ```

 - SegNet

    ```
      ./caffe/build/tools/caffe train --solver models/solver_segnet.prototxt
    ```

    ```
      python ./caffe/build/install/python/train.py --solver models/solver_segnet.prototxt
    ```


 - FCN

     ```
    ./caffe/build/tools/caffe train --solver models/fcn/solver_fcn.prototxt --weights weights/fcn8s-heavy-pascal.caffemodel
    ```


### TEST

    python scripts/test_segmentation.py --model models/segnet_inference.prototxt --weights weights/segnet_iter_50000.caffemodel --test_file dataset/cityscapes_val.txt
    python scripts/test_segnet.py --test_file dataset/cityscapes_val.txt --gpu 0
    python scripts/test_fcn.py --test_file dataset/cityscapes_val.txt --gpu 0


### TOOLS
 - Plot training loss:
 
    python caffe/tools/extra/parse_log.py log_file processed_log

    python scripts/plot_loss.py --log processed_log



**paper: Segmentation and measurement scheme for fish morphological features based on Mask R-CNN**.
[[link](https://www.researchgate.net/publication/338678060_Segmentation_and_Measurement_Scheme_for_Fish_Morphological_Features_Based_on_Mask_R-CNN)]

"train_model.py" is the file that performs the training.
"test_model.py" is the file that executes the test.
"train_data" folder is a training sample set.
"test_result.py" is the folder where the segmentation result graph is saved.
"logs" is the folder that generates the trained model.

**Operating environment**:(CUDA 9.0 cudnn 7.6.5)   
keras 2.1.6   
tensorflow-gpu 1.15.0  
h5py 2.10.0  
numpy  
scipy  
pillow  
cython  
matplotlib  
scikit-image  
opencv-python  
imgaug  
IPython  


When training, change the name of the corresponding study object in "train_model.py".
When testing, change the name of the corresponding study object in "test_model.py".

Before training the model, you should add the pre-training weight parameter file "mask_rcnn_coco.h5", 
create a folder named “image”,"test_result" and "logs" under this project.


If you need data set, go to [[link](https://github.com/Wahaha1314/Fish-characteristic-measurement)]

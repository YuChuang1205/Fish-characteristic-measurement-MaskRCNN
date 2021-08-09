
paper: Segmentation and measurement scheme for fish morphological features based on Mask R-CNN.
[链接](https://www.researchgate.net/publication/338678060_Segmentation_and_Measurement_Scheme_for_Fish_Morphological_Features_Based_on_Mask_R-CNN)
"train_model.py" is the file that performs the training.
"test_model.py" is the file that executes the test.
"train_data" folder is a training sample set.
"test_result.py" is the folder where the segmentation result graph is saved.
"logs" is the folder that generates the trained model.

When training, change the name of the corresponding study object in "train_model.py".
When testing, change the name of the corresponding study object in "test_model.py".

Before training the model, you should add the pre-training weight parameter file "mask_rcnn_coco.h5", 
create a folder named "test_result" and "logs" under this project.

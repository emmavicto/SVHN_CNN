# SVHN_CNN
Building a Concurrent Neural Network classifier for digit recognition using the SVHN dataset.

### Dataset
Character level ground truth in an MNIST-like format. All digits are resized to a fixed resolution of 32-by-32 pixels. The original character bounding boxes are extended in the appropriate dimension to become square windows, so that resizing them to 32-by-32 pixels does not introduce aspect ratio distortions. Nevertheless, this pre-processing introduces some distracting digits to the sides of the digit of interest. 

### Reference model
A classifier built using TFLearn was used as reference. This original solution can be found in https://github.com/codemukul95/SVHN-classification-using-Tensorflow.git

### Pre-Requisites for user
-	opencv-python
-	tflearn
-	Python 3.7.3
- Tensorflow 1.13.1 

### Input data
The data input processing was split into load_test and load_train methods to load the test and train datasets, respectively.
Both methods download the '.mat' files from their respective source URL using urllib.request, if not present in the local directory, and return them in the form of Numpy arrays for the data (X) and Categories for the labels (Y).
Loading the .mat files creates 2 variables: X which is a 4-D matrix containing the images, and y which is a vector of class labels. To access the images, X(:, :, :, i) gives the i-th 32-by-32 RGB image, with class label y(i).

- Train dataset = 73,257 images.
- Test dataset = 26,032 images.
- Validation dataset = 2,000 Images extracted from train. 
- Categories = 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10. 

Labels were converted to categories using tflearn.data_utils, which converts class vectors to binary class matrix, for use with categorical_crossentropy. Input tensors: 4-D Tensors with shape = [batch=None, height=32, width=32, channels=3].
The model has been fed with tensors with a shape of 32 by 32 pixels by 3 channels (RGB). 

### Structure of the svhn.py file

A single file called ‘svhn.py’ has been created containing the following methods:
-	load_train_data & load_test_data -> To download and load train and test datasets, respectively.
-	traintest() -> To trigger the training and testing of the model. It returns the avg F1 score.
-	load_pre_trained_model() -> Used for testing individual images. A pre-trained model is loaded in advance in order to quickly return individual predictions when required.
-	test(test_image) -> To test individual images.

## To Run
You can use the jupyter notebook harness file included or import the svhn class and use the following main functions:

- test(FileName)
Which takes the name of a JPEG or PNG file with a shape of 32-by-32 pixels and returns an integer that corresponds to the most likely
house number seen in the supplied image.

- traintest()
This function when called should (a) download the training and test data, (b) train the model from scratch; and (c) perform analysis against test data. The final output of this function is a production of average F1 scores across each class in the testset.

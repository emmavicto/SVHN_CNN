import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization
import numpy as np
from scipy import io as sio			  
import os.path
import urllib.request					  
from cv2 import cv2
import tensorflow as tf

def load_train_data():

    # Checks if test data exists in the local directory. If not, then downloads it from source repo.
    if not os.path.isfile("train_32x32.mat"):
        print("Downloading Train File")
        with urllib.request.urlopen("http://ufldl.stanford.edu/housenumbers/train_32x32.mat") as response_train , open("train_32x32.mat", 'wb') as train_out_file:
            train_data = response_train.read() 
            train_out_file.write(train_data)
    
    # Converts .mat file to numpy array
    print("Loading Train File")
    train_dict = sio.loadmat("train_32x32.mat")
    X = np.asarray(train_dict['X'])

    X_train = []
    for i in range(X.shape[3]):
        X_train.append(X[:,:,:,i])
    X_train = np.asarray(X_train)

    # Converts labels into Categories
    Y_train = train_dict['y']
    for i in range(len(Y_train)):
        if Y_train[i]%10 == 0:
            Y_train[i] = 0
    Y_train = to_categorical(Y_train[:, 0],10)
    
    # Returns train dataset
    return (X_train,Y_train)

def load_test_data():

    # Checks if train data exists in the local directory. If not, then downloads it from source repo.
    if not os.path.isfile("test_32x32.mat"):
        print("Downloading Test File")
        with urllib.request.urlopen("http://ufldl.stanford.edu/housenumbers/test_32x32.mat") as response_test, open("test_32x32.mat", 'wb') as test_out_file:
            test_data = response_test.read() 
            test_out_file.write(test_data)
    
    # Converts .mat file to numpy array
    print("Downloading Test File")
    test_dict = sio.loadmat("test_32x32.mat")
    X = np.asarray(test_dict['X'])

    X_test = []
    for i in range(X.shape[3]):
        X_test.append(X[:,:,:,i])
    X_test = np.asarray(X_test)

    # Converts labels into Categories
    Y_test = test_dict['y']
    for i in range(len(Y_test)):
        if Y_test[i]%10 == 0:
            Y_test[i] = 0
    Y_test = to_categorical(Y_test[:, 0],10)
    
    # Returns train dataset    
    return (X_test,Y_test)

def traintest():
    # This function builds and trains a model and saves it in the local path.

    # Loading & Shuffling Train data
    X_train, Y_train = load_train_data()
    X_train, Y_train = shuffle(X_train, Y_train)
    print('shuffle done')

    # Define Validation dataset
    X_val = X_train[2000:4000]
    Y_val = Y_train[2000:4000]
    
    tf.reset_default_graph() # Reset graph that could have been imported by the function 'test(test_image)'

    print("Building the CNN architecture")
    network = input_data(shape=[None, 32, 32, 3]) # Input Data Layer

    network = conv_2d(network, 16, 3, activation='relu', weights_init='xavier', name='CN1') # Convolutional Net 1
    network = batch_normalization(network)

    network = conv_2d(network, 16, 3, activation='relu', weights_init='xavier', name='CN2') # Convolutional Net 2
    network = max_pool_2d(network, 2, name='MaxPool1') # Pooling 1 
    network = batch_normalization(network)

    network = conv_2d(network, 32, 3, activation='relu', weights_init='xavier', name='CN3') # Convolutional Net 3
    network = max_pool_2d(network, 2, name='MaxPool2') # Pooling 2
    network = batch_normalization(network)

    network = conv_2d(network, 32, 3, activation='relu', weights_init='xavier', name='CN4') # Convolutional Net 4
    network = max_pool_2d(network, 2, name='MaxPool3') # Pooling 3
    network = batch_normalization(network)

    network = conv_2d(network, 64, 3, activation='relu', weights_init='xavier', name='CN5') # Convolutional Net 5
    network = max_pool_2d(network, 2, name='MaxPool4') # Pooling 4
    network = batch_normalization(network)

    network = fully_connected(network, 256, activation='relu', weights_init='xavier', name='FC1') # Fully Conn Layer 1
    network = dropout(network, 0.25) # dropout 

    network = fully_connected(network, 10, activation='softmax', weights_init='xavier', name='FC2') # Fully Conn Layer 2

    # Defining the Estimator
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    # Creating the DNN model
    model = tflearn.DNN(network, tensorboard_verbose=0)

    # Fitting the model
    print("Fitting the model")
    model.fit(X_train, Y_train, n_epoch=10, shuffle=True, validation_set=(X_val, Y_val),
              show_metric=True, batch_size=100,
              snapshot_epoch=True,
              run_id='svhn_EV_20')

    # Saving the model
    model.save("svhn_model_ev.tfl")
    print("SVHN model trained and saved as svhn_model_ev_new.tfl")
	
	# TEST PHASE
    print("Starging Test phase")
	# Loading & Shuffling Test data
    X_test, Y_test = load_test_data()
    X_test, Y_test = shuffle(X_test, Y_test)

    # Calculate lenght of the test dataset
    total_samples = len(X_test)
    correct_predict = 0.
    
    # Predict label for each image in the set
    for i in range(len(X_test)):
        prediction = model.predict([X_test[i]])
        digit = np.argmax(prediction)
        label = np.argmax(Y_test[i])

        # Sum total true predictions
        if digit == label:
            correct_predict += 1

    # Return accuracy
    return correct_predict / total_samples

def load_pre_trained_model():
    # This function is used for unit testing of individual images. 

    # Loading the model
    network = input_data(shape=[None, 32, 32, 3])

    network = conv_2d(network, 16, 3, activation='relu', weights_init='xavier', name='CN1')
    network = batch_normalization(network)

    network = conv_2d(network, 16, 3, activation='relu', weights_init='xavier', name='CN2')
    network = max_pool_2d(network, 2, name='MaxPool1')
    network = batch_normalization(network)

    network = conv_2d(network, 32, 3, activation='relu', weights_init='xavier', name='CN3')
    network = max_pool_2d(network, 2, name='MaxPool2')
    network = batch_normalization(network)

    network = conv_2d(network, 32, 3, activation='relu', weights_init='xavier', name='CN4')
    network = max_pool_2d(network, 2, name='MaxPool3')
    network = batch_normalization(network)

    network = conv_2d(network, 64, 3, activation='relu', weights_init='xavier', name='CN5')
    network = max_pool_2d(network, 2, name='MaxPool4')
    network = batch_normalization(network)

    network = fully_connected(network, 256, activation='relu', weights_init='xavier', name='FC1')
    network = dropout(network, 0.25)

    network = fully_connected(network, 10, activation='softmax', weights_init='xavier', name='FC2')

    network = regression(network, optimizer='adam',
                        loss='categorical_crossentropy',
                        learning_rate=0.001)

    model = tflearn.DNN(network, tensorboard_verbose=0)

    model.load("svhn_model_ev.tfl")

    return model

# Pre-Loads the pre-trained model. Used to avoid loading the model more than once. Useful for unit testing more than 1 image.
MODEL_TRAINED= load_pre_trained_model()

def test(test_image,model=MODEL_TRAINED):
    # This function accepts an image of a digit in a 32-by-32 pixels RGB format and predicts and return an integer representing the digit in the image.
    # It also receives the pre-loaded model

    converted_image = cv2.imread(test_image) # Converts from .png/.jpeg/.jpe/etc to .mat
    prediction = model.predict([converted_image]) # Predicts the label 
    digit = np.argmax(prediction) 
    
    return digit
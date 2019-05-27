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

    if not  os.path.isfile("train_32x32.mat"):
        with urllib.request.urlopen("http://ufldl.stanford.edu/housenumbers/train_32x32.mat") as response_train , open("train_32x32.mat", 'wb') as train_out_file:
            train_data = response_train.read() 
            train_out_file.write(train_data)
    
    train_dict = sio.loadmat("train_32x32.mat")
    X = np.asarray(train_dict['X'])

    X_train = []
    for i in range(X.shape[3]):
        X_train.append(X[:,:,:,i])
    X_train = np.asarray(X_train)

    Y_train = train_dict['y']
    for i in range(len(Y_train)):
        if Y_train[i]%10 == 0:
            Y_train[i] = 0
    Y_train = to_categorical(Y_train[:, 0],10)
    
    return (X_train,Y_train)

def load_test_data():

    if not os.path.isfile("test_32x32.mat"):
        with urllib.request.urlopen("http://ufldl.stanford.edu/housenumbers/test_32x32.mat") as response_test, open("test_32x32.mat", 'wb') as test_out_file:
            test_data = response_test.read() 
            test_out_file.write(test_data)
    
    test_dict = sio.loadmat("test_32x32.mat")
    X = np.asarray(test_dict['X'])

    X_test = []
    for i in range(X.shape[3]):
        X_test.append(X[:,:,:,i])
    X_test = np.asarray(X_test)

    Y_test = test_dict['y']
    for i in range(len(Y_test)):
        if Y_test[i]%10 == 0:
            Y_test[i] = 0
    Y_test = to_categorical(Y_test[:, 0],10)
    
    return (X_test,Y_test)

def traintest():
    X_train, Y_train = load_train_data()
    X_train, Y_train = shuffle(X_train, Y_train)
    print('shuffle done')

    X_val = X_train[2000:4000]
    Y_val = Y_train[2000:4000]
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

    model = tflearn.DNN(network, tensorboard_verbose=1)

    # tensorboard --logdir=C:\tmp\tflearn_logs

    model.fit(X_train, Y_train, n_epoch=10, shuffle=True, validation_set=(X_val, Y_val),
              show_metric=True, batch_size=100,
              snapshot_epoch=True,
              run_id='svhn_EV_18')

    model.save("svhn_model_ev.tfl")
    print("SVHN model trained and saved as svhn_model_ev.tfl")

    X_test, Y_test = load_test_data()
    X_test, Y_test = shuffle(X_test, Y_test)

    total_samples = len(X_test)
    correct_predict = 0.
    for i in range(len(X_test)):
        prediction = model.predict([X_test[i]])
        digit = np.argmax(prediction)
        label = np.argmax(Y_test[i])

        if digit == label:
            correct_predict += 1

    return correct_predict / total_samples

def load_pre_trained_model():
    if not os.path.isfile("svhn_model_ev.tfl.meta"):
        traintest()

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

    model = tflearn.DNN(network, tensorboard_verbose=1)

    model.load("svhn_model_ev.tfl")

    return model

MODEL_TRAINED= load_pre_trained_model()

def test(test_image,model=MODEL_TRAINED):

    converted_image = cv2.imread(test_image)
    prediction = model.predict([converted_image])
    digit = np.argmax(prediction)
    return digit

    




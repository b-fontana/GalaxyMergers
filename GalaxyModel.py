from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras import backend
from tensorflow.python.keras.losses import categorical_crossentropy
#from tensorflow.python.keras.losses import binary_crossentropy
#from tensorflow.python.keras.optimizers import Adadelta
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import TensorBoard

import os, sys, glob, argparse
import numpy as np
from PIL import Image

from Modules.Data import Data as BData
from Modules.ArgParser import add_args
    
def train(filenames, img_rows, img_cols, extension):
    """
    Trains a model using Keras.
    Expects numpy arrays with values between 0 and 255.
    """
    num_classes, epochs, batch_size = 2, 20, 32
    x, y = GalaxyData.load_from_tfrecord(filenames)

    print("TRAIN: x[0] shape is", x.shape[0])
    print("TRAIN: x shape is", x.shape)
    print("TRAIN: y shape is", y.shape[0])
    
    if backend.image_data_format() == 'channels_first':
        x = x.reshape(x.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x = x.reshape(x.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)
        
    x_train, y_train, x_test, y_test = GalaxyData.split_data(x, y, fraction=0.8)
    #x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.
    print(x_train[5].shape)
    #This is a check
    pic = x_train[5]
    print("Max:", np.amax(x_train[5]))
    print("Min:", np.amin(x_train[5]))
    im = Image.fromarray((x_train[5]*255).astype('uint8'))
    im.save("im_numpy.jpg")
    

    print('x_train shape:', x_train.shape, "; y_train shape", y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices (one-hot encoding)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    print(y_test.shape)
    print(y_train.shape)
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',
                     data_format=backend.image_data_format(),
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
    #model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])

    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              shuffle=True,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5),
                         EarlyStopping(monitor='loss', min_delta=0.00001, patience=5)
                         #ModelCheckpoint(FLAGS.save_model_name, verbose=1, period=1),
                         #TensorBoard(log_dir=FLAGS.tensorboard, batch_size=32)
                     ])

    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save(FLAGS.save_model_name)


def predict(picture_names, height, width):
    """
    Return the predictions for the input_pictures.
    """
    model = load_model(FLAGS.saved_model_name)
    print("Model being used:", FLAGS.saved_model_name)
    picture_array = np.zeros((len(picture_names), height, width, 3), dtype=np.float32)
    dims = (height, width, 3)
    graph, nameholder, image_tensor = picture_decoder((300, 300, 3))
    print(image_tensor.shape)
    with tf.Session(graph=graph) as sess:
        init = tf.group( tf.global_variables_initializer(), tf.local_variables_initializer() )
        sess.run(init)
        for i,name in enumerate(picture_names):
            picture_array[i] = sess.run(image_tensor, feed_dict={nameholder: name})/255.
    return model.predict(picture_array/255., verbose=0)


    
def main(_):
    """
    Code written by Bruno Alves on July 2018.
    Used resources:
    1. Decoding a picture using tensorflow (retraining tensorflow example):
    https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py
    2. Writing to and reading from TFRecord files: 
    http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
    3. Tensorflow, Keras and Python online documentation
    """
    ###Parsed arguments checks###
    if FLAGS.mode == "save" or (FLAGS.mode == "train" and FLAGS.use_saved_data == 0):
        data_extension = os.path.splitext(FLAGS.save_data_name)[1]
        if data_extension != ".tfrecord":
            print("The extensions of the file name inserted could not be accepted.")
            sys.exit()
        if FLAGS.use_saved_data != 0 and FLAGS.use_saved_data != 1:
            print("The 'use_saved_data' is a boolean. Only the '0' and '1' values can be accepted.")
            sys.exit()
        if FLAGS.cutmin >= FLAGS.cutmax:
            print("The 'cutmin' option has to be smaller than the 'cutmax' option.")
            sys.exit()

    elif FLAGS.mode == "train":
        if FLAGS.saved_data_name == None:
            print("Please provide the name of the file(s) where the pictures will be loaded from.")
            sys.exit()
        if FLAGS.saved_data_name != None:
            for filename in FLAGS.saved_data_name:
                data_extension = os.path.splitext(filename)[1]
                if data_extension != ".tfrecord":
                    print("The extensions of the file name(s) inserted could not be accepted.")
                    sys.exit()
        data_extension = os.path.splitext(FLAGS.save_model_name)[1]
        if data_extension != ".h5":
            print("The extension of the model name inserted could not be accepted.")
            sys.exit()

    elif FLAGS.mode == "predict":
        data_extension = os.path.splitext(FLAGS.saved_model_name)[1]
        if data_extension != ".h5" and FLAGS.mode != "save":
            print("The extension of the model name inserted could not be accepted.")
            sys.exit()


    ###Print all passed arguments as a check###
    print()
    print("############Arguments##Info######################")
    print("Mode:", FLAGS.mode)
    print("Data to convert:", FLAGS.data_to_convert)
    print("Saved data name:", FLAGS.saved_data_name)
    print("Save data name:", FLAGS.save_data_name)
    print("Saved model name:", FLAGS.saved_model_name)
    print("Save model name:", FLAGS.save_model_name)
    print("Tensorboard:", FLAGS.tensorboard)
    print("Cutmin:", FLAGS.cutmin)
    print("Cutmax:", FLAGS.cutmax)
    print("#################################################")
    print()

    GalaxyData = BData("galaxies")

    img_rows, img_cols = 300, 300
    ###Saving the data if requested###
    if FLAGS.use_saved_data == 0 and (FLAGS.mode == 'train' or FLAGS.mode == 'save'):
        myclasses = ('before', 'during')
        mypath = "/data1/alves/"+FLAGS.data_to_convert
        GalaxyData.save_to_tfrecord(mypath, myclasses, 
                                    FLAGS.save_data_name, 
                                    FLAGS.cutmin, FLAGS.cutmax, 
                                    img_rows, img_cols, 3, 'jpg')
    
    ###Training or predicting###
    if FLAGS.mode == 'train':
        train(FLAGS.saved_data_name, img_rows, img_cols, 'jpg')
    elif FLAGS.mode == 'predict':
        if not os.path.isfile(FLAGS.saved_model_name):
            print("The saved model could not be found.")
            sys.exit()
        pict_names = ['twoballs.jpg',
                      '/data1/alves/galaxy_photos_balanced_bckg/before/a9_outFile-00300-04.jpg',
                      '/data1/alves/galaxy_photos_balanced_bckg/before/b52_outFile-00330-11.jpg',
                      '/data1/alves/galaxy_photos_balanced_bckg/during/b71_outFile-00450-09.jpg',
                      '/data1/alves/galaxy_photos_balanced_bckg/during/a7_outFile-00500-06.jpg']
        result = predict(pict_names, img_rows, img_cols)
        for i in range(len(pict_names)):
            print("Prediction:", result[i])
    else: #if the mode is 'save' there is nothing else to do
        if FLAGS.mode != 'save':
            print("The specified mode is not supported. \n Currently two options are supported: 'train' and 'predict'.")
        sys.exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, unparsed = add_args(parser)
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
    sys.exit()

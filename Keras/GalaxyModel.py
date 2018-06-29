from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras import backend
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.optimizers import Adadelta

import os
import glob
import gc
import numpy as np
from PIL import Image


def picture_decoder(tf_session, picture_name, height, width):
    """
    Part of a tensorflow graph that converts ('decodes') a picture into a tensor and performs
    other extra image manipulations
    
    Arguments:
    1. picture_name: path of the picture one whishes to decode
    2. height, width and depth: dimensions of the final decoded picture
    
    Returns:
    1. decoded picture (numpy ndarray)
    """
    picture_name_tensor = tf.placeholder( tf.string )
    picture_contents = tf.read_file( picture_name_tensor )

    picture =  tf.image.decode_jpeg( picture_contents )                   
    picture_as_float = tf.image.convert_image_dtype( picture, tf.float32 )
    picture_4d = tf.expand_dims( picture_as_float, 0 )
 
    resize_shape = tf.stack([height, width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)

    final_tensor =  tf.image.resize_bilinear( picture_4d, resize_shape_as_int )
    return tf_session.run( final_tensor, feed_dict={picture_name_tensor: picture_name} )


def get_data(DataFolder, Classes, Height=300, Width=300, Extension='jpg'):
    """
    Creates two arrays: one with pictures and the other with numerical labels
    Each class must have a separate subfolder, and all classes must lie inside the same data folder.

    Arguments:
    1. DataFolder (string): folder where all the pictures from all the classes are stored.
    2. Classes (tuple): classes to be considered.
    3. Extension (string): extension of the pictures inside DataFolder. Only 'jpg' is supported. 

    Returns:
    1. A 4d array with pictures and another array with labels. The first array follows the following format: (index, height_in_pixels, width_in_pixels, numer_of_channels)
    2. The given classes are converted into numerical labels, starting from zero. For example, if three classes are present, 'a', 'b' and 'c', then the labels with respectively be 0, 1 and 2.
    """

    ###Sanity checks###
    allowed_extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    if Extension not in allowed_extensions:
        print("ERROR: The", Extension, "extension is not supported.")
        quit()
    if not os.path.exists(DataFolder):
        print("ERROR: The specified", DataFolder, "folder does not exist.")
        quit()
    for nameClass in Classes:
        if not os.path.exists( os.path.join(DataFolder,nameClass) ):
            print("ERROR: The specified", os.path.join(DataFolder,nameClass), "does not exist." )
            quit()

    ###Tensorflow Picture Decoding###
    data_tuple = ()
    input_height, input_width, input_depth = Height, Width, 3
    image_pattern_list = []
    temp_number = 0

    pict_array_length = np.zeros(len(Classes), dtype=np.uint32)
    for iClass,nameClass in enumerate(Classes):
        pict_array_length[iClass]=len(glob.glob(os.path.join(DataFolder,nameClass,"*."+Extension))[:50])
 
    pict_array_length_total = sum(pict_array_length)
    pict_array = np.zeros((pict_array_length_total, input_width, input_width, input_depth), 
                          dtype=np.float32)
    label_array = np.zeros(pict_array_length_total, dtype=np.uint8)

    with tf.Session() as sess:
        init = tf.group( tf.global_variables_initializer(), tf.local_variables_initializer() )
        sess.run(init)

        for iClass, nameClass in enumerate(Classes):
            for i,pict in enumerate(glob.glob(os.path.join(DataFolder,nameClass,"*."+Extension))[:50]):
                index = i + iClass*pict_array_length[iClass]
                if index%20 == 0:
                    print(index+1, "pictures have been decoded", 
                          float(index)/float(pict_array_length_total)*100, "%")
                #ndarray with dtype=float32
                pict_array[index] = picture_decoder(sess, pict, input_height, input_width) 
                label_array[index] = iClass
    
    print("Shape of the array of pictures:", pict_array.shape)
    print("Shape of the array of labels:", label_array.shape)
    return (pict_array, label_array)


def split_data(x, y, fraction=0.8):
    """
    Splits the data into 'training' and 'testing' datasets according to the specified fraction.

    Arguments:
    1. The actual data values ('x')
    2. The label array ('y')
    3. The fraction of training data the user wants

    Returns:
    The testing and training data and labels in the following order:
    (x_train, y_train, x_test, y_test)
    """
    ###Sanity Check###
    if len(x) != len(y):
        print("ERROR: The arrays of the values and of the labels must have the same size!")
        quit()
    
    splitting_value = int(len(x)*fraction)
    return x[:splitting_value], y[:splitting_value], x[splitting_value:], y[splitting_value:]



def main(_):
    myclasses = ('before', 'during')
    mypath = "/data1/alves/galaxy_photos_balanced_gap/"
    img_rows, img_cols = 300, 300
    num_classes, epochs, batch_size = 2, 10, 64
    x, y = get_data(mypath, myclasses, img_rows, img_cols, 'jpg')

    if backend.image_data_format() == 'channels_first':
        x = x.reshape(x.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x = x.reshape(x.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)
        
    x_train, y_train, x_test, y_test = split_data(x, y, fraction=0.8)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape, "; y_train shape", y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices (one-hot encoding)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',
                     data_format=backend.image_data_format(),
                     input_shape=input_shape))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    tf.app.run(main=main)

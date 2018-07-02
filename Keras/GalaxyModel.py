from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras import backend
#from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.losses import binary_crossentropy
#from tensorflow.python.keras.optimizers import Adadelta
from tensorflow.python.keras.callbacks import EarlyStopping

import os, sys, glob, argparse
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


def save_data(DataFolder, Classes, Height=300, Width=300, Depth=3, Extension='jpg'):
    """
    Creates two arrays: one with pictures and the other with numerical labels
    Each class must have a separate subfolder, and all classes must lie inside the same data folder.

    Arguments:
    1. DataFolder (string): folder where all the pictures from all the classes are stored.
    2. Classes (tuple): classes to be considered.
    3. Extension (string): extension of the pictures inside DataFolder. Only 'jpg' is supported. 

    Stores:
    1. A 4d array with pictures and another array with labels. The first array follows the following format: (index, height_in_pixels, width_in_pixels, numer_of_channels)
    2. The given classes are converted into numerical labels, starting from zero. For example, if three classes are present, 'a', 'b' and 'c', then the labels with respectively be 0, 1 and 2.

    Returns: nothing
    """
    ###Functions to be used for storing the files as TFRecords###
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    ###Sanity checks###
    allowed_extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    if Extension not in allowed_extensions:
        print("ERROR: The", Extension, "extension is not supported.")
        sys.exit()
    if not os.path.exists(DataFolder):
        print("ERROR: The specified", DataFolder, "folder does not exist.")
        sys.exit()
    for nameClass in Classes:
        if not os.path.exists( os.path.join(DataFolder,nameClass) ):
            print("ERROR: The specified", os.path.join(DataFolder,nameClass), "does not exist." )
            sys.exit()

    ###Tensorflow Picture Decoding###
    input_height, input_width, input_depth = Height, Width, Depth
    with tf.Session() as sess:
        init = tf.group( tf.global_variables_initializer(), tf.local_variables_initializer() )
        sess.run(init)
        
        #Define a counter (the saving takes a while...)
        pict_array_length = np.zeros(len(Classes), dtype=np.uint32)
        for iClass, nameClass in enumerate(Classes):
            for i,pict in enumerate(glob.glob(os.path.join(DataFolder,nameClass,"*."+Extension))[:200]):
                pict_array_length[iClass] = len(glob.glob(os.path.join(DataFolder,nameClass,"*."+Extension))[:200])
                pict_array_length_total =+ pict_array_length[iClass]

        #Write to a .tfrecord file
        Writer = tf.python_io.TFRecordWriter(FLAGS.saved_data_name)
        for iClass, nameClass in enumerate(Classes):
            for i,pict in enumerate(glob.glob(os.path.join(DataFolder,nameClass,"*."+Extension))[:200]):
                index = i + iClass*pict_array_length[iClass]
                if index%20 == 0:
                    print(index+1, "pictures have been decoded", 
                          float(index)/float(pict_array_length_total)*100, "%")
                tmp_picture = picture_decoder(sess, pict, input_height, input_width) #ndarray with dtype=float32 
                
                Example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(input_height),
                    'width': _int64_feature(input_width),
                    'depth': _int64_feature(input_depth),
                    'picture_raw': _bytes_feature(tmp_picture.tostring()),
                    'label': _int64_feature(iClass)}))
    
                Writer.write(Example.SerializeToString())
        Writer.close()    
        print("Shape of a single saved picture:", tmp_picture.shape)

def load_data(filename):
    """
    Loads a TFRecords binary file containing pictures information and converts it back to numpy array format. The function does not know before-hand how many pictures are stored in 'filename'.
    
    Arguments:
    1. The name of the file to load.
    
    Returns:
    1. A 4d array with pictures and another array with labels. The first array follows the following format: (index, height_in_pixels, width_in_pixels, numer_of_channels)
    2. The given classes are converted into numerical labels, starting from zero. For example, if three classes are present, 'a', 'b' and 'c', then the labels with respectively be 0, 1 and 2.
    """
    if not os.path.isfile(filename):
        print("The data could not be loaded. Please make sure the filename is correct.")
        sys.exit()
        
    pict_array, label_array = ([] for i in range(2)) #such a fancy initialization!
    iterator = tf.python_io.tf_record_iterator(path=filename)

    for i, element in enumerate(iterator):
        Example = tf.train.Example()
        Example.ParseFromString(element)
        height = int(Example.features.feature['height'].int64_list.value[0])
        width = int(Example.features.feature['width'].int64_list.value[0])
        depth = int(Example.features.feature['depth'].int64_list.value[0])
        img_string = (Example.features.feature['picture_raw'].bytes_list.value[0])
        pict_array.append( np.fromstring(img_string, dtype=np.float32).reshape((height,width,depth)) )
        label_array.append( (Example.features.feature['label'].int64_list.value[0]) )
        #if i%20 == 0:
         #           print(i+1, "pictures have been loaded",
          #                float(i)/float(len(iterator))*100, "%")

    pict_array = np.array(pict_array)
    label_array = np.array(label_array)
    print("Shape of the loaded array of pictures:", pict_array.shape)
    print("Shape of the loaded array of labels:", label_array.shape)
    return (pict_array, label_array)


    ###If I decide to create my own model with tensors, this is the way to go###
    ###With Keras this makes not much sense:###
    ###I would have to convert binary files into tensors and then back to numpy arrays###
    """
    tf_filename = tf.placeholder(tf.string)
    dataset = tf.data.TFRecordDataset(tf_filename)
    dataset = dataset.map(   )
    dataset = dataset.repeat()
    dataset = dataset.batch(32)
    iterator = dataset.make_initializable_iterator() #there are other kinds of iterators
    training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
    validation_filenames = ["/var/data/validation1.tfrecord", ...]
    sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
    """


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
        sys.exit()
    
    splitting_value = int(len(x)*fraction)
    return x[:splitting_value], y[:splitting_value], x[splitting_value:], y[splitting_value:]



def train(filename, img_rows, img_cols, extension):
    """
    Trains a model
    """
    num_classes, epochs, batch_size = 2, 100, 32
    x, y = load_data(filename)

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

    print(y_train)
    print('x_train shape:', x_train.shape, "; y_train shape", y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices (one-hot encoding)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(128, kernel_size=(5, 5), activation='relu',
                     data_format=backend.image_data_format(),
                     input_shape=input_shape))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0002, patience=5),
                         EarlyStopping(monitor='loss', min_delta=0.0002, patience=5)])

    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('keras_model.h5')


def predict(picture_names, heigth, width):
    """
    Return the predictions for the input_pictures.
    """
    model = load_model('keras_model.h5')

    picture_array = np.zeros((5, heigth, width, 3), dtype=np.float32)
    with tf.Session() as sess:
        for i,name in enumerate(picture_names):
            picture_array[i] = picture_decoder(sess, name, heigth, width)

    return model.predict(picture_array, verbose=1)


    
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
    data_extension = os.path.splitext(FLAGS.saved_data_name)[1]
    if data_extension != ".h5" and data_extension != ".tfrecord":
        print("The extensions of the file name(s) inserted could not be accepted.")
        sys.exit()
    if FLAGS.use_saved_data != 0 and FLAGS.use_saved_data != 1:
        print("The 'use_saved_data' is a boolean. Only the '0' and '1' values can be accepted.")
        sys.exit()

    img_rows, img_cols = 300, 300
    ###Saving the data if requested###
    if FLAGS.use_saved_data == 0:
        myclasses = ('before', 'during')
        mypath = "/data1/alves/galaxy_photos_balanced_gap/"
        print("yes")
        save_data(mypath, myclasses, img_rows, img_cols, 3, 'jpg')

    ###Training or predicting###
    if FLAGS.mode == 'train':
        train(FLAGS.saved_data_name, img_rows, img_cols, 'jpg')
    elif FLAGS.mode == 'predict':
        if not os.path.isfile(FLAGS.saved_model_name):
            print("The saved model could not be found.")
            sys.exit()
        pict_names = ['/data1/alves/galaxy_photos_balanced_gap/before/c95_outFile-00300-12.jpg',
                      '/data1/alves/galaxy_photos_balanced_gap/before/b49_outFile-00305-04.jpg',
                      '/data1/alves/galaxy_photos_balanced_gap/before/c88_outFile-00330-06.jpg',
                      '/data1/alves/galaxy_photos_balanced_gap/during/b72_outFile-00450-06.jpg',
                      '/data1/alves/galaxy_photos_balanced_gap/during/d144_outFile-00480-09.jpg']
        result = predict(pict_names, img_rows, img_cols)
        for i in range(len(pict_names)):
            print("Prediction (before):", result[i])
    else:
        print("The specified mode is not supported. \n Currently two options are supported: 'train' and 'predict'.")
        sys.exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_saved_data',
        type=int,
        default=0,
        help='Imports (or not) the TFRecords file containing previously saved data. This saves a lot of computational time. Default: 0'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='Mode. Currently available options: \n 1) train \n 2) predict'
    )
    parser.add_argument(
        '--saved_model_name',
        type=str,
        default='keras_model.h5',
        help="Name of the file where the model is going to be saved. It must have a 'h5' extension."
    )
    parser.add_argument(
        '--saved_data_name',
        type=str,
        default='galaxy_pictures.tfrecord',
        help="Name of the file where the data is going to be saved. It must have a 'tfrecord' extension."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
    sys.exit()

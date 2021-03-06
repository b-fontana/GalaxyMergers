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
import time

#os.environ['CUDA_VISIBLE_DEVICES']="1,2,3"

class LoopInfo:
    """Handles all loop-related informations
    
    Arguments:
    1. total_len: number of items the loop will process'
    """
    def __init__(self, total_len=-1):
        self.total_len = total_len
        self.time_init = time.time()

    def loop_print(self, it, step_time):
        """
        The time is measured from the last loop_info call
        """
        subtraction = step_time - self.time_init
        if self.total_len == -1:
            print("{0:d} iteration. Iteration time: {1:.3f}" .format(it, subtraction))
        else:
            percentage = (float(it)/self.total_len)*100
            print("{0:.2f}% finished. Iteration time: {1:.3f}" .format(percentage, subtraction))
        self.time_init = time.time()    


def picture_decoder(dims):
    """
    Graph that decodes a jpeg image.

    Returns:
    1. The graph
    2. The tensor which provides the decoded picture in its final form
    3. The placeholder for the picture name
    """
    g = tf.Graph()
    with g.as_default():
        picture_name_tensor = tf.placeholder(tf.string)
        picture_contents = tf.read_file(picture_name_tensor)
        picture = tf.image.decode_jpeg(picture_contents, dct_method="INTEGER_ACCURATE")
        picture_as_float = tf.image.convert_image_dtype(picture, tf.float32)*255
        picture_4d = tf.expand_dims(picture_as_float, 0)
        resize_shape = tf.stack([dims[0], dims[1]])
        resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
        final_tensor =  tf.image.resize_bilinear(picture_4d, resize_shape_as_int, 
                                                 align_corners=True)
    return g, picture_name_tensor, final_tensor


def picture_decoder_numpy(picture_name, height, width):
    """
    Converts a picture to an array using numpy.
    Equivalent to the 'picture_decoder' funcion, but much faster, and I don't know why!
    """
    image = Image.open(picture_name)
    image = image.resize((height,width), Image.LANCZOS)
    image = np.array(image, dtype=np.int32) #the number can be at most 256
    return np.expand_dims(image, axis=0)
    

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
    if not os.path.isdir(DataFolder):
        print("ERROR: The specified", DataFolder, "folder does not exist.")
        sys.exit()
    for nameClass in Classes:
        if not os.path.exists( os.path.join(DataFolder,nameClass) ):
            print("ERROR: The specified", os.path.join(DataFolder,nameClass), "does not exist." )
            sys.exit()

    ###Tensorflow Picture Decoding###
    dim_tuple = (Height, Width, Depth)
    graph, nameholder, image_tensor = picture_decoder(dim_tuple)
    with tf.Session(graph=graph) as sess:
        init = tf.group( tf.global_variables_initializer(), tf.local_variables_initializer() )
        sess.run(init)
        
        indiv_len = np.zeros(len(Classes), dtype=np.uint32)
        for iClass, nameClass in enumerate(Classes):
            glob_list = glob.glob(os.path.join(DataFolder,nameClass,"*."+Extension))
            indiv_len[iClass]=len(glob_list[FLAGS.cutmin:FLAGS.cutmax])
        total_len = sum(indiv_len)

        #Write to a .tfrecord file
        loop = LoopInfo(total_len)
        counter=0
        with tf.python_io.TFRecordWriter(FLAGS.save_data_name) as Writer:
            for iClass, nameClass in enumerate(Classes):
                glob_list = glob.glob(os.path.join(DataFolder,nameClass,"*."+Extension))[FLAGS.cutmin:FLAGS.cutmax]
                print(len(glob_list))
                for i,pict in enumerate(glob_list):
                    index = i + iClass*indiv_len[iClass-1] if iClass != 0 else i 
                    tmp_picture = sess.run(image_tensor, feed_dict={nameholder: pict} )
                    #tmp_picture = picture_decoder_numpy(pict, dim_tuple[0], dim_tuple[1])
                    if index%100 == 0:
                        loop.loop_print(index, time.time())
                    Example = tf.train.Example(features=tf.train.Features(feature={
                        'height': _int64_feature(dim_tuple[0]),
                        'width': _int64_feature(dim_tuple[1]),
                        'depth': _int64_feature(dim_tuple[2]),
                        'picture_raw': _bytes_feature(tmp_picture.tostring()),
                        'label': _int64_feature(iClass)
                    }))
                    Writer.write(Example.SerializeToString())
                    counter += 1
        print("The data was saved.", counter)
        sys.exit()
def load_data(filenames):
    """
    Loads one or more TFRecords binary file(s) containing pictures information and converts it/them back to numpy array format. 
    The function does not know before-hand how many pictures are stored in 'filenames'.
    
    Arguments:
    1. The names of the files to load.
    
    Returns:
    1. A 4d array with pictures and another array with labels. The first array follows the following format: (index, height_in_pixels, width_in_pixels, numer_of_channels)
    2. The given classes are converted into numerical labels, starting from zero. For example, if three classes are present, 'a', 'b' and 'c', then the labels with respectively be 0, 1 and 2.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            print("The data stored in", filename, 
                  "could not be loaded. Please make sure the filename is correct.")
            sys.exit()
        
    pict_array, label_array = ([] for i in range(2)) #such a fancy initialization!
    for filename in filenames:
        iterator = tf.python_io.tf_record_iterator(path=filename)
        loop = LoopInfo()
        for i, element in enumerate(iterator):
            Example = tf.train.Example()
            Example.ParseFromString(element)
            height = int(Example.features.feature['height'].int64_list.value[0])
            width = int(Example.features.feature['width'].int64_list.value[0])
            depth = int(Example.features.feature['depth'].int64_list.value[0])
            img_string = (Example.features.feature['picture_raw'].bytes_list.value[0])
            pict_array.append(np.fromstring(img_string,dtype=np.float32).reshape((height,width,depth)))
            label_array.append( (Example.features.feature['label'].int64_list.value[0]) )
            if i%500 == 0:
                loop.loop_print(i,time.time())
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


def unison_shuffle(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


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
    
    ###Shuffling###
    unison_shuffle(x, y)
    
    splitting_value = int(len(x)*fraction)
    return x[:splitting_value], y[:splitting_value], x[splitting_value:], y[splitting_value:]


def train(filenames, img_rows, img_cols, extension):
    """
    Trains a model using Keras.
    Expects numpy arrays with values between 0 and 255.
    """
    num_classes, epochs, batch_size = 2, 20, 32
    x, y = load_data(filenames)

    print("TRAIN: x[0] shape is", x.shape[0])
    print("TRAIN: x shape is", x.shape)
    print("TRAIN: y shape is", y.shape[0])
    
    if backend.image_data_format() == 'channels_first':
        x = x.reshape(x.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x = x.reshape(x.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)
        
    x_train, y_train, x_test, y_test = split_data(x, y, fraction=0.8)
    #x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
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
            print(i)
            picture_array[i] = sess.run(image_tensor, feed_dict={nameholder: name})
            
    print(model.predict(picture_array, verbose=1))


    
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

    start_time = time.time()
    img_rows, img_cols = 300, 300
    ###Saving the data if requested###
    if FLAGS.use_saved_data == 0 and (FLAGS.mode == 'train' or FLAGS.mode == 'save'):
        myclasses = ('before', 'during')
        mypath = "/data1/alves/"+FLAGS.data_to_convert
        save_data(mypath, myclasses, img_rows, img_cols, 3, 'jpg')
    
    ###Training or predicting###
    if FLAGS.mode == 'train':
        train(FLAGS.saved_data_name, img_rows, img_cols, 'jpg')
    elif FLAGS.mode == 'predict':
        if not os.path.isfile(FLAGS.saved_model_name):
            print("The saved model could not be found.")
            sys.exit()
        pict_names = ['mug.jpg',
                      '/data1/alves/blur+contours/galaxy_photos_balanced_bckg/before/a9_outFile-00300-04_blur+contour.jpg',
                      '/data1/alves/blur+contours/galaxy_photos_balanced_bckg/before/b52_outFile-00330-11_blur+contour.jpg',
                      '/data1/alves/blur+contours/galaxy_photos_balanced_bckg/during/b71_outFile-00450-09_blur+contour.jpg',
                      '/data1/alves/blur+contours/galaxy_photos_balanced_bckg/during/a7_outFile-00500-06_blur+contour.jpg']
        predict(pict_names, img_rows, img_cols)
        #for i in range(len(pict_names)):
         #   print("Prediction:", result[i])
    else: #if the mode is 'save' there is nothing else to do
        if FLAGS.mode != 'save':
            print("The specified mode is not supported. \n Currently two options are supported: 'train' and 'predict'.")
        sys.exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_saved_data',
        type=int,
        help='Imports (or not) the TFRecords file containing previously saved data. This saves a lot of computational time. Default: 0'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='Mode. Currently available options: \n 1) save \n 2) train \n 3) predict'
    )
    parser.add_argument(
        '--save_model_name',
        type=str,
        help="Name of the file where the model is going to be saved. It must have a 'h5' extension."
    )
    parser.add_argument(
        '--data_to_convert',
        type=str,
        help="Name of the folder where the pictures to be converted to the '.tfrecord' format are stored. It is assumed that that folder can be found in /data1/alves/. Note that the classes have to be stored in different folders inside the specified 'data_to_convert' folder. By default, the two classes being conisdered are 'before' and 'during'."
    )
    parser.add_argument(
        '--saved_model_name',
        type=str,
        help="Name of the file where the model was saved. It must have a 'h5' extension."
    )
    parser.add_argument(
        '--save_data_name',
        type=str,
        help="File name where the data is going to be saved. It must have a 'tfrecord' extension."
    )
    parser.add_argument(
        '--saved_data_name', 
        nargs='+', 
        type=str,
        help="File name(s) where the data was saved. They must have a 'tfrecord' extension."
    )
    parser.add_argument(
        '--cutmin',
        type=int,
        default=0,
        help='Left range of the array of pictures to be saved for each class.'
    )
    parser.add_argument(
        '--cutmax',
        type=int,
        default=999999,
        help='Right range of the array of pictures to be saved for each class.'
    )
    parser.add_argument(
        '--tensorboard',
        type=str,
        help='Right range of the array of pictures to be saved for each class.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
    sys.exit()

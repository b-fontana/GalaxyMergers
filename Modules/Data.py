import os, glob, sys, time
import tensorflow as tf
import numpy as np

from Modules.Infos import LoopInfo
from Modules.General import unison_shuffle
from Modules.Picture import Picture

class Data:
    'Class for handling data operations'
    def __init__(self, name=""):
        self.name = name
        
        
    def save_to_tfrecord(self, DataFolder, Classes, DataName, DataMin=0, DataMax=99999999, Height=300, Width=300, Depth=3, Extension='jpg'):
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
        Pics = Picture()
        graph, nameholder, image_tensor = Pics.tf_decoder(Height, Width)
        with tf.Session(graph=graph) as sess:
            init = tf.group( tf.global_variables_initializer(), tf.local_variables_initializer() )
            sess.run(init)
            indiv_len = np.zeros(len(Classes), dtype=np.uint32)
            for iClass, nameClass in enumerate(Classes):
                glob_list = glob.glob(os.path.join(DataFolder,nameClass,"*."+Extension))
                indiv_len[iClass]=len(glob_list[DataMin:DataMax])
                total_len = sum(indiv_len)

            #Write to a .tfrecord file
            loop = LoopInfo(total_len)
            with tf.python_io.TFRecordWriter(DataName) as Writer:
                for iClass, nameClass in enumerate(Classes):
                    glob_list = glob.glob(os.path.join(DataFolder,nameClass,"*."+Extension))
                    glob_list = glob_list[DataMin:DataMax]
                    for i,pict in enumerate(glob_list):
                        index = i + iClass*indiv_len[iClass-1] if iClass != 0 else i
                        tmp_picture = sess.run(image_tensor, feed_dict={nameholder: pict} )
                        #tmp_picture = Pics.np_decoder(pict, Height, Width)               
                        if index%100 == 0:
                            loop.loop_print(index, time.time())
                        Example = tf.train.Example(features=tf.train.Features(feature={
                            'height': _int64_feature(Height),
                            'width': _int64_feature(Width),
                            'depth': _int64_feature(Depth),
                            'picture_raw': _bytes_feature(tmp_picture.tostring()),
                            'label': _int64_feature(iClass)
                        }))
                        Writer.write(Example.SerializeToString())
        print("The data was saved.")


    def load_from_tfrecord(self, filenames, class_number, dims):
        """
        Converts TFRecord files into a TensorFlow Dataset. 
  
        Arguments: The names of the files to load
        Returns: A mapped tf.data.Dataset (pictures, labels)
        """
        for filename in filenames:
            if not os.path.isfile(filename):
                print("The data stored in", filename,
                      "could not be loaded. Please make sure the filename is correct.")
                sys.exit()

        def parser_func(tfrecord):
            features = {'picture_raw': tf.FixedLenFeature((), tf.string),
                        'label': tf.FixedLenFeature((), tf.int64)}
            parsed_features = tf.parse_single_example(tfrecord, features)

            picture = tf.decode_raw(parsed_features['picture_raw'], tf.float32)
            picture /= 255.
            #picture_shape = tf.cast(parsed_features['height'], tf.int64),
            #                 tf.cast(parsed_features['width'], tf.int64),
            #                 tf.cast(parsed_features['depth'], tf.int64)
            picture = tf.reshape(picture, [dims[0], dims[1], dims[2]])
            label = parsed_features['label']
            label = tf.one_hot(indices=label, depth=class_number, on_value=1, off_value=0)
            label = tf.cast(label, tf.float32)
            return picture, label

        dataset = tf.data.TFRecordDataset(filenames)
        return dataset.map(parser_func)


    def split_data(self, dataset, ntest):
        test_dataset = dataset.take(ntest)
        train_dataset = dataset.skip(ntest)
        return train_dataset, test_dataset

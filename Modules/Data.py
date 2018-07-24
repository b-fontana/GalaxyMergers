import os, glob, sys, time
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import to_categorical

from Modules.Infos import LoopInfo
from Modules.General import unison_shuffle
from Modules.Picture import Picture

class Data:
    'Class for handling data operations'
    def __init__(self, name=""):
        self.name = name
        
        
    def save_tfrec(self, DataFolder, Classes, DataName, Dims, DataMin=0, DataMax=99999999, Extension='jpg'):
        """
        Saves data in the TFRecord format. The data is stored in a structured way, with one folder per class.

        Arguments:
        1. DataFolder (string): folder where all the pictures from all the classes are stored.
        2. Classes (tuple): classes to be considered
        3. DataName (string): name of the file where the data is going to be stored
        4. Dims (tuple): dimensions of the data pictures
        5. DataMin (int): Minimum index of a list representing the full dataset
        6. DataMax (int): Maximum index of a list representing the full dataset
        7. Extension (string): extension of the pictures inside DataFolder. Only 'jpg' is supported.
        
        Stores:
        TFRecord file containing the pictures and their labels (0, 1, 2 ...)
        """
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
        graph, nameholder, image_tensor = Pics.tf_decoder(Dims)
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
                        if index%100 == 0:
                            loop.loop_print(index, time.time())
                        Example = tf.train.Example(features=tf.train.Features(feature={
                            'picture_raw': _bytes_feature(tmp_picture.tostring()),
                            'label': _int64_feature(iClass)
                        }))
                        Writer.write(Example.SerializeToString())
        print("The data was saved.")

    
    def save_tfrec_bonsai(self, FileNames, DataName, Dims, Extension):
        """
        Save bonsai data in the TFRecord format
        
        Arguments:
        1. FileNames (string): files (full path) where the data is stored
        2. DataName (string): name of the TFRecord file
        3. Dims (tuple of integers): dimensions of the pictures (heigth, width, depth)
        4. Extension (string): extension of the data pictures

        Stores:
        TFRecord file with the pictures and their parameters' indices (mass_ratio, mass_size, ...)
        """
        import ctypes
        _ = ctypes.CDLL('/home/alves/Clibraries/jsmn/libjsmn2.so', mode=ctypes.RTLD_GLOBAL)
        lib = ctypes.CDLL('/home/alves/Clibraries/libreader.so')

        param_names = ['vr', 'vt', 'vt_phi', 'size_ratio', 'mass_ratio', 'Rsep',
                       'lMW', 'bMW', 'lM31', 'bM31', 'lR', 'bR']
        param_values = [[-130., -90., -50.],                    #vr
                        [10., 20., 30.],                        #vt
                        [-45., 0., 45.],                        #vt_phi
                        [0.25, 0.5, 0.75, 1., 1.25, 1.5],       #size_ratio
                        [0.25, 0.5, 0.75, 1., 1.25, 1.5],       #mass_ratio
                        [778., 788., 798.],                     #Rsep
                        [0., 90., 180.],                        #lMW
                        [-90., 0., 90.],                        #bMW
                        [200., 220., 240.],                     #lM31
                        [-90., -60., -30.],                     #bM31
                        [120., 121., 122.],                     #lR
                        [-25., -23., -21.]]                     #bR
  
        ###Functions to be used for storing the files as TFRecords###
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def parameter_idx(param, value):
            idx = param_names.index(param)
            return param_values[idx].index(value)

        Loop = LoopInfo()
        Pics = Picture()
        graph, nameholder, image_tensor = Pics.tf_decoder(Dims)
        with tf.Session(graph=graph) as sess:
            init = tf.group( tf.global_variables_initializer(), tf.local_variables_initializer() )
            sess.run(init)
            for iFileName,FileName in enumerate(FileNames):
                pic_path = os.path.join(FileName, "bonsai_simulations/s_*")
                folders = glob.glob(pic_path)
                
                for i_folder, folder in enumerate(folders):

                    json_file_name = os.path.join(FileName, "bonsai_simulations", 
                                                  folder, "params.json")
                    bufstr = ctypes.create_string_buffer(bytes(json_file_name, encoding='utf-8'))
                    lib.get_json_bonsai_reader_size.restype = ctypes.c_int
                    reader_size = lib.get_json_bonsai_reader_size()
                    parameters = np.zeros(reader_size, dtype=np.float32)
                    lib.json_bonsai_reader.restype = ctypes.c_void_p
                    lib.json_bonsai_reader(ctypes.c_char_p(ctypes.addressof(bufstr)), 
                                           ctypes.c_void_p(parameters.ctypes.data))
                    pics = glob.glob(os.path.join(folder,"*."+Extension))
            
                    split1, split2 = DataName.split('.') 
                    folder_name = split1+str(i_folder)+"_"+str(iFileName)+"."+split2
                    
                    with tf.python_io.TFRecordWriter(folder_name) as Writer:
                        for i_pic, pic in enumerate(pics):
                            pic_raw = sess.run(image_tensor, feed_dict={nameholder: pic} )
                            Example = tf.train.Example(features=tf.train.Features(feature={
                                'picture_raw': _bytes_feature(pic_raw.tostring()),
                                'vr_idx': _int64_feature(parameter_idx('vr', parameters[0])),
                                'vt_idx': _int64_feature(parameter_idx('vt', parameters[1])),
                                'vt_phi_idx': _int64_feature(parameter_idx('vt_phi', parameters[2])),
                                'size_ratio_idx': _int64_feature(parameter_idx('size_ratio',parameters[3])),
                                'mass_ratio_idx': _int64_feature(parameter_idx('mass_ratio',parameters[4])),
                                'Rsep_idx': _int64_feature(parameter_idx('Rsep', parameters[5])),
                                'lMW_idx': _int64_feature(parameter_idx('lMW', parameters[6])),
                                'bMW_idx': _int64_feature(parameter_idx('bMW', parameters[7])),
                                'lM31_idx': _int64_feature(parameter_idx('lM31', parameters[8])),
                                'bM31_idx': _int64_feature(parameter_idx('bM31', parameters[9])),
                                'lR_idx': _int64_feature(parameter_idx('lR', parameters[10])),
                                'bR_idx': _int64_feature(parameter_idx('bR', parameters[11]))
                            }))
                            Writer.write(Example.SerializeToString())


    def load_tfrec(self, filenames, class_number, dims):
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
            picture = tf.reshape(picture, [dims[0], dims[1], dims[2]])
            label = parsed_features['label']
            label = tf.one_hot(indices=label, depth=class_number, on_value=1, off_value=0)
            label = tf.cast(label, tf.float32)
            return picture, label

        dataset = tf.data.TFRecordDataset(filenames)
        return dataset.map(parser_func)

    def load_tfrec_bonsai(self, filenames, dims):
        """
        Converts TFRecord files into a TensorFlow Dataset with one-hot-encoded labels. 
  
        Arguments: The names of the files to load
        Returns: A mapped tf.data.Dataset (pictures, labels)
        """
        for filename in filenames:
            if not os.path.isfile(filename):
                print("The data stored in", filename,
                      "could not be loaded. Please make sure the filename is correct.")
                sys.exit()

        param_names = ['vr', 'vt', 'vt_phi', 'size_ratio', 'mass_ratio', 'Rsep',
                       'lMW', 'bMW', 'lM31', 'bM31', 'lR', 'bR']
        param_values = [[-130., -90., -50.],                    #vr
                        [10., 20., 30.],                        #vt
                        [-45., 0., 45.],                        #vt_phi
                        [0.25, 0.5, 0.75, 1., 1.25, 1.5],       #size_ratio
                        [0.25, 0.5, 0.75, 1., 1.25, 1.5],       #mass_ratio
                        [778., 788., 798.],                     #Rsep
                        [0., 90., 180.],                        #lMW
                        [-90., 0., 90.],                        #bMW
                        [200., 220., 240.],                     #lM31
                        [-90., -60., -30.],                     #bM31
                        [120., 121., 122.],                     #lR
                        [-25., -23., -21.]]                     #bR

        def get_len(param):
            idx = param_names.index(param)
            return len(param_values[idx])

        def parser_func(tfrecord):
            """
            def get_list_len(idx):
                return len(param_list[idx])

            def get_index(par, value):
                idx = parameters.index(par)
                return param_list[idx].index(value)
            """
            features = {'picture_raw': tf.FixedLenFeature((), tf.string)}
            for param in param_names:
                features.update({param+"_idx": tf.FixedLenFeature((), tf.int64)})

            parsed_features = tf.parse_single_example(tfrecord, features)

            picture = tf.decode_raw(parsed_features['picture_raw'], tf.float32)
            picture /= 255.
            picture = tf.reshape(picture, [dims[0], dims[1], dims[2]])
            
            features_final=tf.stack([#tf.one_hot(parsed_features['vr_idx'],3),
                                     #tf.one_hot(parsed_features['vt_idx'],3)])
                                     #tf.one_hot(parsed_features['vt_phi_idx']),3]
                                     tf.one_hot(parsed_features['size_ratio_idx'], get_len('size_ratio')),
                                     tf.one_hot(parsed_features['mass_ratio_idx'], get_len('mass_ratio'))])
                                     #tf.one_hot(parsed_features['Rsep_idx')],3]
                                     #tf.one_hot(parsed_features['lMW_idx')],3]
                                     #tf.one_hot(parsed_features['bMW_idx')],3]
                                     #tf.one_hot(parsed_features['lM31_idx')],3]
                                     #tf.one_hot(parsed_features['bM31_idx')],3]
                                     #tf.one_hot(parsed_features['lR_idx')],3]
                                     #tf.one_hot(parsed_features['bR_idx')],3]
            features_final = tf.reshape(features_final, [-1])
            return picture, features_final

        dataset = tf.data.TFRecordDataset(filenames)
        return dataset.map(map_func=parser_func, 
                           num_parallel_calls=192)


    ###Split training and testing dataset
    def split_data(self, dataset, ntest):
        test_dataset = dataset.take(ntest)
        train_dataset = dataset.skip(ntest)
        return train_dataset, test_dataset


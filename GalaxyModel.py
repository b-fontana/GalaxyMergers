from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras import layers 
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras import backend
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import TensorBoard

import os, sys, glob, argparse
import numpy as np
from PIL import Image

from Modules.Data import Data as BData
from Modules.Picture import Picture as BPic
from Modules.ArgParser import add_args

os.environ["CUDA_VISIBLE_DEVICES"]="1"
    
def nn_sequential(inputs, shape, nclass):
    """
    Creates and returns neural net model
    """
    x = layers.Conv2D(32, kernel_size=(3, 3), 
                          activation='relu',
                          padding='valid',
                          data_format=backend.image_data_format(),
                          input_shape=shape)(inputs)
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                      padding='valid')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    #x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu',
                      #padding='valid')(x)
    #x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    #x = layers.Dense(256, activation='relu')(x)
    #x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(nclass, activation='softmax')(x)
    return x

def train(filenames, dims, extension):
    """
    Trains a model using Keras.
    Expects numpy arrays with values between 0 and 255.
    """
    nclasses, nepochs, batch_size = 2, 10, 64
    npics = 0
    for filename in filenames:
        for record in tf.python_io.tf_record_iterator(filename):
            npics += 1

    dataset = BData().load_from_tfrecord(filenames, nclasses, dims)
    dataset = dataset.shuffle(buffer_size=npics)
    dataset = dataset.repeat(nepochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    
    x, y = iterator.get_next()
    
    if backend.image_data_format() == 'channels_first':
        input_shape = (depth, img_rows, img_cols)
    else:
        input_shape = (dims[0], dims[1], dims[2])

        model_input = layers.Input(tensor=x, shape=(dims[0], dims[1], dims[2]))
        model_output = nn_sequential(model_input, input_shape, nclasses)
        model = Model(inputs=model_input, outputs=model_output)
        model.compile(optimizer='adam',
                      loss=categorical_crossentropy,
                      metrics=['accuracy'],
                      target_tensors=[y])    
        model.summary()
        print("STEPS_PER_EPOCH:", int((npics+batch_size-1)/batch_size))
        print("NPICS:", npics)
        print("BATCH_SIZE:", batch_size)
        model.fit(shuffle=True,
                  epochs=nepochs,
                  steps_per_epoch=int((npics+batch_size-1)/batch_size),
                  verbose=1,
                  callbacks=[EarlyStopping(monitor='loss', min_delta=0.00001, patience=5),
                             #ModelCheckpoint(FLAGS.save_model_name, verbose=1, period=1),
                             TensorBoard(log_dir=FLAGS.tensorboard, batch_size=batch_size)])

        model.save(FLAGS.save_model_name)


def predict(picture_names, dims):
    """
    Return the predictions for the input_pictures.
    """
    model = load_model(FLAGS.saved_model_name)
    print("Model being used:", FLAGS.saved_model_name)
    picture_array = np.zeros((len(picture_names), dims[0], dims[1], dims[2]), dtype=np.float32)
    graph, nameholder, image_tensor = BPic().tf_decoder(dims)

    with tf.Session(graph=graph) as sess:
        init = tf.group( tf.global_variables_initializer(), tf.local_variables_initializer() )
        sess.run(init)
        for i,name in enumerate(picture_names):
            picture_array[i] = sess.run(image_tensor, feed_dict={nameholder: name})
            picture_array[i] = np.array(picture_array[i], dtype=np.float32)
#            picture_array[i] = picture_array[i].astype('float32') 
            picture_array[i] /= 255
            
        print(picture_array[0])
    print(np.amax(picture_array[0]))
    print(np.amin(picture_array[0]))

    print(model.predict(picture_array, verbose=0))

    
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
        if FLAGS.tensorboard == None:
            print("Please specify the tensorboard folder.")
            sys.exit()
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
    print("RGB:", FLAGS.input_depth)
    print("#################################################")
    print()

    GalaxyData = BData("galaxies")

    dims_tuple = (300, 300, FLAGS.input_depth)
    ###Saving the data if requested###
    if FLAGS.use_saved_data == 0 and (FLAGS.mode == 'train' or FLAGS.mode == 'save'):
        myclasses = ('before', 'during')
        mypath = "/data1/alves/"+FLAGS.data_to_convert
        GalaxyData.save_to_tfrecord(mypath, myclasses, 
                                    FLAGS.save_data_name,
                                    dims_tuple,
                                    FLAGS.cutmin, FLAGS.cutmax, 'jpg')
    
    ###Training or predicting###
    if FLAGS.mode == 'train':
        train(FLAGS.saved_data_name, dims_tuple, 'jpg')
    elif FLAGS.mode == 'predict':
        if not os.path.isfile(FLAGS.saved_model_name):
            print("The saved model could not be found.")
            sys.exit()
        predict(prediction_list(), dims_tuple)
    else: #if the mode is 'save' there is nothing else to do
        if FLAGS.mode != 'save':
            print("The specified mode is not supported. \n Currently two options are supported: 'train' and 'predict'.")
        sys.exit()

def prediction_list():
    return ['/data1/alves/GalaxyZoo/noninteracting/training_587738947752099924.jpeg',
            '/data1/alves/GalaxyZoo/noninteracting/test_587724650336485508.jpeg',
            '/data1/alves/GalaxyZoo/noninteracting/test_587722982297633240.jpeg',
            '/data1/alves/GalaxyZoo/noninteracting/training_587729160042119287.jpeg',
            '/data1/alves/GalaxyZoo/noninteracting/training_588013384353382565.jpeg',
            '/data1/alves/GalaxyZoo/noninteracting/validation_588297864188133564.jpeg',
            '/data1/alves/GalaxyZoo/noninteracting/training_587729160044675203.jpeg',  
            '/data1/alves/GalaxyZoo/noninteracting/training_588013384356986950.jpeg',  
            '/data1/alves/GalaxyZoo/noninteracting/validation_588297864189247559.jpeg',
            '/data1/alves/GalaxyZoo/noninteracting/training_587729160048279578.jpeg',
            '/data1/alves/GalaxyZoo/noninteracting/training_588013384357314676.jpeg',
            '/data1/alves/GalaxyZoo/noninteracting/validation_588297864724021378.jpeg',
            '/data1/alves/GalaxyZoo/noninteracting/training_587729160048345212.jpeg',  
            '/data1/alves/GalaxyZoo/noninteracting/training_588015507653853241.jpeg',
            '/data1/alves/GalaxyZoo/noninteracting/validation_588297865250472179.jpeg',
            '/data1/alves/GalaxyZoo/noninteracting/training_587729160049066132.jpeg',

            '/data1/alves/GalaxyZoo/merger/training_588015507655819397.jpeg',
            '/data1/alves/GalaxyZoo/merger/validation_588023239133495463.jpeg',
            '/data1/alves/GalaxyZoo/merger/training_587728932419862622.jpeg',
            '/data1/alves/GalaxyZoo/merger/training_588015507660996743.jpeg',  
            '/data1/alves/GalaxyZoo/merger/validation_588023239133495464.jpeg',
            '/data1/alves/GalaxyZoo/merger/training_587728932956012731.jpeg',  
            '/data1/alves/GalaxyZoo/merger/training_588015507660996744.jpeg',  
            '/data1/alves/GalaxyZoo/merger/validation_588023239671873686.jpeg',
            '/data1/alves/GalaxyZoo/merger/training_587728932956012732.jpeg',  
            '/data1/alves/GalaxyZoo/merger/training_588015507664928990.jpeg',  
            '/data1/alves/GalaxyZoo/merger/validation_588023239671873687.jpeg',
            '/data1/alves/GalaxyZoo/merger/training_587728949052506235.jpeg',  
            '/data1/alves/GalaxyZoo/merger/training_588015507664928991.jpeg',  
            '/data1/alves/GalaxyZoo/merger/validation_588295842319630367.jpeg',
            '/data1/alves/GalaxyZoo/merger/training_587728949052506236.jpeg',
            '/data1/alves/GalaxyZoo/merger/training_588015507680723008.jpeg']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, unparsed = add_args(parser)
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
    sys.exit()

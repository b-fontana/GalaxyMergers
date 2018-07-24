from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras import layers 
from tensorflow.python.keras import backend
#from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import TensorBoard

import os, sys, glob, argparse
import numpy as np
from PIL import Image

from Modules.Data import Data as BData
from Modules.Picture import Picture as BPic
from Modules.ArgParser import add_args
from Modules.Callbacks import Testing

os.environ["CUDA_VISIBLE_DEVICES"]="3"
    
def nn(inputs, shape, nclass):
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
    #x = layers.Dense(nclass, activation='softmax')(x)
    x = layers.Dense(nclass, activation='sigmoid')(x)
    return x

def train(filenames, dims, extension):
    """
    Trains a model using Keras.
    Expects numpy arrays with values between 0 and 255.
    """
    nclasses, nepochs, batch_size = 12, 50, 192
    fraction = 0.8 #training fraction
    npics = 0
    for filename in filenames:
        for record in tf.python_io.tf_record_iterator(filename):
            npics += 1
    steps_per_epoch = int((npics+batch_size-1)/batch_size)

    print()
    print("STEPS_PER_EPOCH:", steps_per_epoch)
    print()

    dataset = BData().load_tfrec_bonsai(filenames, dims)
    dataset = dataset.shuffle(buffer_size=npics)
    train_dataset, test_dataset = BData().split_data(dataset, int(npics*(1-fraction)))
    train_dataset = train_dataset.repeat(nepochs+1)
    train_dataset = train_dataset.batch(batch_size)
    train_iterator = train_dataset.make_one_shot_iterator()

    x_train, y_train = train_iterator.get_next()
    print("y_train shape:", y_train.shape)

    if backend.image_data_format() == 'channels_first':
        input_shape = (dims[2], dims[0], dims[1])
    else:
        input_shape = (dims[0], dims[1], dims[2])

    train_callback = Testing(test_dataset, int(npics*(1-fraction)), batch_size)

    model_input = layers.Input(tensor=x_train, shape=input_shape)
    model_output = nn(model_input, input_shape, nclasses)
    model = Model(inputs=model_input, outputs=model_output)

    model.compile(optimizer='adam',
                  #loss=categorical_crossentropy,
                  loss=binary_crossentropy, #squash labels between 0 and 1 for using sigmoid
                  metrics=['accuracy'],
                  target_tensors=[y_train])
    model.summary()
    model.fit(shuffle=True,
              epochs=nepochs,
              steps_per_epoch=steps_per_epoch,
              verbose=1,
              callbacks=[EarlyStopping(monitor='loss', min_delta=0.000005, patience=5),
                         ModelCheckpoint(FLAGS.save_model_name, verbose=1, period=1),
                         TensorBoard(log_dir=FLAGS.tensorboard, batch_size=batch_size),
                         train_callback])
    model.save(FLAGS.save_model_name)


def predict(picture_names, dims):
    """
    Return the predictions for the input_pictures.
    """
    param_names = ['vr', 'vt', 'vt_phi', 'size_ratio', 'mass_ratio', 'Rsep',
                   'lMW', 'bMW', 'lM31', 'bM31', 'lR', 'bR']
    param_values = [[-130., -90., -50.],               #vr
                    [10., 20., 30.],                   #vt
                    [-45., 0., 45.],                   #vt_phi
                    [0.25, 0.5, 0.75, 1., 1.25, 1.5],  #size_ratio
                    [0.25, 0.5, 0.75, 1., 1.25, 1.5],  #mass_ratio
                    [778., 788., 798.],                #Rsep
                    [0., 90., 180.],                   #lMW
                    [-90., 0., 90.],                   #bMW
                    [200., 220., 240.],                #lM31
                    [-90., -60., -30.],                #bM31
                    [120., 121., 122.],                #lR
                    [-25., -23., -21.]]                #bR

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
            picture_array[i] /= 255
    predictions = model.predict(picture_array, verbose=0).flatten()
    max1 = np.amax(predictions[:6])
    max2 = np.amax(predictions[6:])
    print(predictions)
    idx1 = np.where(predictions==max1)[0][0]
    idx2 = np.where(predictions==max2)[0][0]-6
    print("I think the size ratio is", param_values[param_names.index('size_ratio')][idx1])
    print("I think the mass ratio is", param_values[param_names.index('mass_ratio')][idx2])

    
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
    if FLAGS.mode == "save":
        data_extension = os.path.splitext(FLAGS.save_data_name)[1]
        if data_extension != ".tfrecord":
            print("The extensions of the file name inserted could not be accepted.")
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
    if FLAGS.mode == 'save':
        print(FLAGS.data_to_convert)
        GalaxyData.save_tfrec_bonsai(FLAGS.data_to_convert, 
                                     FLAGS.save_data_name, 
                                     dims_tuple,
                                     'jpg')
    elif FLAGS.mode == 'train':
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
    """return ['/data1/alves/GalaxyZoo/noninteracting/training_587738947752099924.jpeg',
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

    return ['/data1/alves/contours/GalaxyZoo/merger/validation_588023046401818861_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/merger/training_587736753004478741_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/merger/validation_588023046401818862_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/merger/training_587736753540235433_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/merger/validation_588023046939476090_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/merger/training_587736753540235434_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/merger/validation_588023046939476091_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/merger/training_587736781994262752_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/merger/validation_588023046944391390_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/merger/training_587736781994262753_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/merger/validation_588023046944391391_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/merger/training_587736781995573405_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/merger/validation_588023048018395313_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/merger/training_587736781995573406_contour.jpeg',

            '/data1/alves/contours/GalaxyZoo/noninteracting/validation_588297864188133564_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/noninteracting/training_587738568170668109_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/noninteracting/validation_588297864189247559_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/noninteracting/training_587738568174272622_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/noninteracting/validation_588297864724021378_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/noninteracting/training_587738568176369826_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/noninteracting/validation_588297865250472179_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/noninteracting/training_587738568702951549_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/noninteracting/validation_588298662504169504_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/noninteracting/training_587738568705441963_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/noninteracting/validation_588298662505939047_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/noninteracting/training_587738568710160617_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/noninteracting/validation_588848900467392548_contour.jpeg',
            '/data1/alves/contours/GalaxyZoo/noninteracting/training_587738574610497709_contour.jpeg']
    """
    #return ["/data1/LEAPSData/LEAPS1bf/bonsai_simulations/s_0.5_m_0.5_lMW_0_bMW_90/outFile-00315-13.jpg"]
    return ["/data1/LEAPSData/LEAPS1bf/bonsai_simulations/s_0.25_m_0.75_lMW_0_bMW_90/outFile-00310-00.jpg"]
    #return ['mug.jpg']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, unparsed = add_args(parser)
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
    sys.exit()

import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras import backend as K

class Testing(Callback):
    def __init__(self, dataset, ninputs, nbatches):
        """
        Arguments:
        1. The testing dataset 
        2. The number of inputs of the testing dataset
       "
        self.dataset = dataset
        self.ninputs = ninputs
        #self.nbatches = self.params['batch_size']
        #self.nepochs = self.params['epochs']
        #self.nbatches_per_epoch = int((self.ninputs+self.nbatches-1)/self.nbatches)
        self.dataset = self.dataset.shuffle(buffer_size=self.ninputs)
        print()
        print("N_INPUTS:", self.ninputs)
        print()
        self.dataset = self.dataset.repeat(self.nepochs)
        self.dataset = self.dataset.batch(1)
        #self.iterator = dataset.make_initializable_iterator()
        self.iterator = dataset.make_one_shot_iterator()
        self.next_element = self.iterator.get_next()
        """
        self.dataset = dataset
        self.ninputs = ninputs
        self.nbatches = nbatches

    def on_train_begin(self, logs={}):
        self.nepochs = logs.get('epochs')
        self.nbatches_per_epoch = int((self.ninputs+self.nbatches-1)/self.nbatches)

        self.dataset = self.dataset.shuffle(buffer_size=self.ninputs)
        print()
        print("N_INPUTS:", self.ninputs)
        print()
        self.dataset = self.dataset.repeat(self.nepochs)
        self.dataset = self.dataset.batch(1)
       
        self.iterator = self.dataset.make_one_shot_iterator()
        self.next_element = self.iterator.get_next()

    def on_epoch_begin(self, epoch, logs={}):
        #self.session = K.get_session()
        #self.session.run(self.iterator.initializer)
        return
 
    def on_epoch_end(self, epoch, logs={}):
        
        def pretty_print(bpe):
            model_eval, model_metr =  self.model.evaluate(steps=bpe), self.model.metrics_names 
            loss_val = model_eval[0]
            acc_val = model_eval[1]
            loss_name = model_metr[0]
            acc_name = model_metr[1]
            print("---|Validation|---  %s: %f | %s: %f" % (loss_name, loss_val, acc_name, acc_val))
        
        pretty_print(self.nbatches_per_epoch)
        return

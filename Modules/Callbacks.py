import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback

class Testing(Callback):
    def __init__(self, dataset, ninputs, nbatches, filename):
        """
        Arguments:
        1. The testing dataset 
        2. The number of inputs of the testing dataset
        3. The number of batches of he dataset 
              (it is not present in the logs because this number is not 
               defined when feeding tensorflow tensors directly)
        """
        self.ninputs = ninputs
        self.nbatches = nbatches
        self.dataset = dataset
        self.dataset = self.dataset.batch(32)
        self.iterator = self.dataset.make_one_shot_iterator()
        self.next_element = self.iterator.get_next()
        self.filename = filename

    def on_train_begin(self, logs={}):
        self.nepochs = logs.get('epochs')
        self.nbatches_per_epoch = int((self.ninputs+self.nbatches-1)/self.nbatches)-1
        self.f = open(self.filename, 'at')
        
    def on_epoch_end(self, epoch, logs={}):
        
        def file_print(bpe):
            model_eval =  self.model.evaluate(steps=bpe) 
            loss_val = model_eval[0]
            acc_val = model_eval[1]
            self.f.write("%f\t%f\n" % (loss_val, acc_val))
        
        def pretty_print(bpe):
            model_eval, model_metr =  self.model.evaluate(steps=bpe), self.model.metrics_names 
            loss_val = model_eval[0]
            acc_val = model_eval[1]
            loss_name = model_metr[0]
            acc_name = model_metr[1]
            print("---|Validation|---  %s: %f | %s: %f" % 
                  (loss_name, loss_val, acc_name, acc_val))
        
        pretty_print(self.nbatches_per_epoch)
        file_print(self.nbatches_per_epoch)
        return


class WriteTrainMetrics(Callback):
    def __init__(self, filename):
        self.filename = filename

    def on_train_begin(self, logs={}):
        self.f = open(self.filename, 'at') 
        
    def on_epoch_end(self, epoch, logs={}):
        self.f.write("%f\t%f\n" % (logs.get('loss'), logs.get('acc')))

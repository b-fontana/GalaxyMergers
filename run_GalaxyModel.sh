#!/usr/bin/env bash

python GalaxyModel.py --mode train --saved_data_name TFRecords/bckg*.tfrecord --tensorboard Tensorboard/bckg --save_model_name KerasModels/bckg.h5
#python GalaxyModel.py --mode save --save_data_name TFRecords/bckg.tfrecord --data_to_convert /data1/LEAPSData/LEAPS1bf/ /data1/LEAPSData/LEAPS2bf/
#python GalaxyModel.py --mode predict --saved_model_name KerasModels/bckg.h5

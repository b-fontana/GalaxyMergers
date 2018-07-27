#!/usr/bin/env bash

#---TRAIN---#
python GalaxyModel.py --mode train --saved_data_name TFRecords/bckg*.tfrecord --tensorboard Tensorboard/bckg --save_model_name KerasModels/test.h5 --file_train_metrics /home/alves/train_metrics.txt --file_test_metrics /home/alves/test_metrics.txt

#---SAVE---#
#python GalaxyModel.py --mode save --save_data_name TFRecords/bckg_contours.tfrecord --data_to_convert /data1/alves/blur+contours/

#---PREDICT---#
if false; then
    FOLDER_PATH="/data1/LEAPSData/LEAPS4bf/bonsai_simulations/"
    CLASSES=(s_0.25_m_0.25 s_0.25_m_0.5 s_0.25_m_0.75 s_0.25_m_1 s_0.25_m_1.25 s_0.25_m_1.5 s_0.5_m_0.25 s_0.5_m_0.5 s_0.5_m_0.75 s_0.5_m_1 s_0.5_m_1.25 s_0.5_m_1.5 s_0.75_m_0.25 s_0.75_m_0.5 s_0.75_m_0.75 s_0.75_m_1 s_0.75_m_1.25 s_0.75_m_1.5 s_1.25_m_0.25 s_1.25_m_0.5 s_1.25_m_0.75 s_1.25_m_1 s_1.25_m_1.25 s_1.25_m_1.5 s_1.5_m_0.25 s_1.5_m_0.5 s_1.5_m_0.75 s_1.5_m_1 s_1.5_m_1.25 s_1.5_m_1.5 s_1_m_0.25 s_1_m_0.5 s_1_m_0.75 s_1_m_1 s_1_m_1.25 s_1_m_1.5)
    EXTRA="_lMW_-90_bMW_-90"
    FILE="prediction_file.txt"

    for i_class in "${CLASSES[@]}"; do
        if [ ! -d "$FOLDER_PATH$i_class" ]; then
	    for i_pic in "$FOLDER_PATH$i_class$EXTRA/"*".jpg"; do
		tmp="${i_pic%_lMW*}"
		tmp="${tmp#*simulations/s_}"
		sr="${tmp%_m*}"  #size ratio
		mr="${tmp#*_m_}" #mass ratio
		echo "$i_pic $sr $mr" >> prediction_list.txt
	    done
        fi
    done
fi
#python GalaxyModel.py --mode predict --saved_model_name KerasModels/bckg.h5 

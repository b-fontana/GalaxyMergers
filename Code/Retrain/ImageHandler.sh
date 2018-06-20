#!/usr/bin/env bash

TARGET_DIRECTORY="/data1/alves/galaxy_photos_gap/"
declare -a TARGET_SUBDIRECTORY=("before/" "during/" "after/")
BOUNDARY_VALUES=(360 390 510 540)
##BOUNDARY_VALUES=(400 530)

CreateDirectory () {
    if [ -d "$1" ]; then
        if [ -L "$1" ]; then
            echo "This is a symbolic link, not a directory!"
            exit 0
        fi
    else
        mkdir "$1"
        echo "$1 created."
    fi
}

read -n 1 -p "Do you wish to delete the $TARGET_DIRECTORY folder (y/n)?" yn
echo
case $yn in
    [Yy]* ) rm -rf "$TARGET_DIRECTORY";;
    [Nn]* ) exit 0;;
    * ) echo "Please answer yes or no.";;
esac

##Check whether the folder where the data will be stored already exists
CreateDirectory "$TARGET_DIRECTORY"

##Loop through the pictures and store them in their respective subfolder according to some class
declare -a DATA_DIRECTORY=("/data1/LEAPSData/LEAPS1/bonsai_simulations/" 
                           "/data1/LEAPSData/LEAPS2/bonsai_simulations/" 
			   "/data1/LEAPSData/LEAPS3/bonsai_simulations/" 
			   "/data1/LEAPSData/LEAPS4/bonsai_simulations/")
declare -a FOLDER_VALUES=("0.25" "0.5" "0.75" "1" "1.25" "1.5")
DATA_DIRECTORY_COUNTER=0
FOLDER_VALUES_COUNTER=0
for dir in "${DATA_DIRECTORY[@]}"; do
    DATA_DIRECTORY_COUNTER=$(($DATA_DIRECTORY_COUNTER + 1))
    if [ "$DATA_DIRECTORY_COUNTER" -eq 1 ]; then
	EXTRA_CHAR="a"
    elif [ "$DATA_DIRECTORY_COUNTER" -eq 2 ]; then
	EXTRA_CHAR="b"
    elif [ "$DATA_DIRECTORY_COUNTER" -eq 3 ]; then
	EXTRA_CHAR="c"
    else
	EXTRA_CHAR="d"
    fi
    for value1 in "${FOLDER_VALUES[@]}"; do
	for value2 in "${FOLDER_VALUES[@]}"; do
	    FOLDER_VALUES_COUNTER=$(($FOLDER_VALUES_COUNTER + 1))
	    if [ "$DATA_DIRECTORY_COUNTER" -le 2 ]; then
		INTERMEDIATE_DIRECTORY="s_${value1}_m_${value2}/"
	    elif [ "$DATA_DIRECTORY_COUNTER" -eq 3 ]; then #LEAPS3/ subfolder
		INTERMEDIATE_DIRECTORY="s_${value1}_m_${value2}_lMW_90_bMW_90/"
	    else #LEAPS4/ subfolder
		INTERMEDIATE_DIRECTORY="s_${value1}_m_${value2}_lMW_-90_bMW_-90/"
	    fi
	    cd "$dir$INTERMEDIATE_DIRECTORY"
	    for picture in *jpg; do
		#Extract only the relevant number from the name of the picture
		tmp=${picture%-*}
		tmp2=${tmp#*00}
		#Choose the folder to store the picture according to the class
		if [ "$tmp2" -le "${BOUNDARY_VALUES[0]}" ]; then
		    CreateDirectory "$TARGET_DIRECTORY${TARGET_SUBDIRECTORY[0]}"
		    cp -i "${picture}" "$TARGET_DIRECTORY${TARGET_SUBDIRECTORY[0]}$EXTRA_CHAR${FOLDER_VALUES_COUNTER}_$picture" 
		elif [ "$tmp2" -gt "${BOUNDARY_VALUES[1]}" -a "$tmp2" -lt "${BOUNDARY_VALUES[2]}" ]; then		    
		    CreateDirectory "$TARGET_DIRECTORY${TARGET_SUBDIRECTORY[1]}"
                    cp -i "${picture}" "$TARGET_DIRECTORY${TARGET_SUBDIRECTORY[1]}$EXTRA_CHAR${FOLDER_VALUES_COUNTER}_$picture"
		elif [ "$tmp2" -gt "${BOUNDARY_VALUES[3]}" ]; then		     
		    CreateDirectory "$TARGET_DIRECTORY${TARGET_SUBDIRECTORY[2]}"
                    cp -i "${picture}" "$TARGET_DIRECTORY${TARGET_SUBDIRECTORY[2]}$EXTRA_CHAR${FOLDER_VALUES_COUNTER}_$picture"
	        fi
	    done
        done
    done
done
exit 0 

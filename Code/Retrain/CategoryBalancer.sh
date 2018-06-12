#!/usr/bin/env bash 

#parsing section
POSITIONAL=()
while [[ $# -gt 0 ]]; do
key="$1"
case $key in
    -m|--minimum)
    MINIMUM="$2"
    shift 
    shift 
    ;;
    -h|--help)
    echo "Options: -m (number of files to be put in each subfolder)"
    shift
    ;;
    --default)
    DEFAULT=YES
    shift                                                                               
    ;;
    *)    # unknown option                                                                              
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters  

#defining global variables
PICTURE_DIRECTORY="$HOME/Data/galaxy_photos2/"
declare -a PICTURE_SUBDIRECTORY=("before/" "during/" "after/")
TARGET_DIRECTORY="$HOME/Data/galaxy_photos_balanced2/"

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

GetArrayMinimum () {
local MINIMUM=9999999
for i in "$@"; do
    if [ "$i" -lt "$MINIMUM" ]; then
	MINIMUM="$i"
    fi
    done
echo "$MINIMUM"
}

#count the number of files in each category folder ('subdirectory') and get the minimum
declare -a FILE_NUMBER=()
for dir in "${PICTURE_SUBDIRECTORY[@]}"; do
    tmp=$( ls "$PICTURE_DIRECTORY$dir" | wc -l )
    FILE_NUMBER+=("$tmp")  
    done
#otherwise MINIMUM was parsed by the user
if [ -z "$MINIMUM" ]; then 
    MINIMUM="$( GetArrayMinimum "${FILE_NUMBER[@]}" )"
fi


#move pictures to new folder such that all the folders have an equal number of pictures
read -n 1 -p "Do you wish to delete the $TARGET_DIRECTORY folder (y/n)?" yn
echo
case $yn in
    [Yy]* ) rm -rf "$TARGET_DIRECTORY";;
    [Nn]* ) exit 0;;
    * ) echo "Please answer yes or no.";;
esac

CreateDirectory "$TARGET_DIRECTORY"
for subdir in "${PICTURE_SUBDIRECTORY[@]}"; do
    CreateDirectory "$TARGET_DIRECTORY$subdir"
    ls "$PICTURE_DIRECTORY$subdir" | shuf -n "$MINIMUM" | xargs -i cp -i "$PICTURE_DIRECTORY$subdir"{} "$TARGET_DIRECTORY$subdir"
    done

echo "Files moved. Each category has now $MINIMUM files."
exit 0

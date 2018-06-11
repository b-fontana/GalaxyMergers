#!/usr/bin/env bash 

PICTURE_DIRECTORY="$HOME/Data/galaxy_photos/"
declare -a PICTURE_SUBDIRECTORY=("before/" "during/" "after/")
TARGET_DIRECTORY="$HOME/Data/galaxy_photos_balanced/"
declare -a TARGET_SUBDIRECTORY=("before/" "during/" "after/")

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
    echo "$dir"
    tmp=$( ls "$PICTURE_DIRECTORY$dir" | wc -l )
    FILE_NUMBER+=("$tmp")  
    done
MINIMUM="$( GetArrayMinimum "${FILE_NUMBER[@]}" )"

#move pictures to new folder such that all the folders have an equal number of pictures
if [ -d "$TARGET_DIRECTORY" ]; then
    read -n 1 -p "Do you wish to delete the $TARGET_DIRECTORY folder (y/n)?" yn
    echo
    case $yn in
	[Yy]* ) rm -rf "$TARGET_DIRECTORY";;
	[Nn]* ) exit 0;;
	* ) echo "Please answer yes or no.";;
    esac
fi

CreateDirectory "$TARGET_DIRECTORY"
for subdir in "${TARGET_SUBDIRECTORY[@]}"; do
    CreateDirectory "$subdir"
    ls "$PICTURE_DIRECTORY$PICTURE_SUBDIRECTORY" | shuf -n "$MINIMUM" | xargs -i mv {} "$subdir"
    done

echo "Files moved. Each category has now $MINIMUM files."



    

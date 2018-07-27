import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter
from pylab import *

import os
import glob
import copy
import sys
sys.path.append( os.environ['HOME'] )
from Modules.General import isListEmpty, var_range

#############################
####PARSING##################
#############################
import argparse
from argparse import RawTextHelpFormatter
parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument(
    '--type',
    type=str,
    default='contours',
    help='Type of picture manipulation that will be performed. The current available options are:\n1) contours\n2) blur\n3) blur+contours'
)
parser.add_argument(
      '--dir_extension',
      type=str,
      default='',
      help='Extension to the name of the folder where the pictures will be stored. This avoids overwriting previous pictures when the same --type option is specified in different ocasions'
)
parser.add_argument(
    '--blur_radius',
    type=str,
    default='4',
    help='Radius of the gaussian blur applied to the picture. It is only valid when the blur type is selected.'
)
parser.add_argument(
    '--classes',
    nargs='+',
    type=str,
    help='Classes array.'
)
ARGS, unparsed = parser.parse_known_args()
if (ARGS.type != "contours" and ARGS.type != "blur" and ARGS.type != "blur+contours"):
    print("Please provide a valid type of picture manipulation thorugh the '--type' option.")
    quit()
if ( (ARGS.type != "blur" and ARGS.type != "blur+contours") and ARGS.blur_radius != "2"):
    print("The blur_radius option can only be used when the chosen type includes blurring. You have selected the", ARGS.type, "type.")
    quit()
if (len(ARGS.classes)<2):
    print("Please introduce at least two classes.")
    print(ARGS.classes)
    quit()

################################################
###GET PICTURE ARRAY AND PERFORM CHECKS#########
################################################
home_save_dir = "/data1/alves/"
home_load_dir = "/data1/LEAPSData/"
#image_save_dir = ""
image_load_dir = ["LEAPS1bf/bonsai_simulations/","LEAPS2bf/bonsai_simulations/"]
target_dir = ARGS.type + ARGS.dir_extension #this directory should already exist
if not os.path.isdir( os.path.join(home_save_dir, target_dir) ):
    print("The specified target directory,", os.path.join(home_save_dir, target_dir), "does not exist. Please create one.")
    quit()
for _, _, files in os.walk( os.path.join(home_save_dir, target_dir), topdown = False ):
    if files:
        print("The specified target directory is not empty. Please correct this.")
        quit()
extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

classes = ARGS.classes
classes_init = copy.copy(classes)

def class_naming_expansion(classes, number_of_folders):
    classes_init = copy.copy(classes)
    for i in range(number_of_folders):
        if i > 0: 
            classes.extend(classes_init)
        for i_class in range(len(classes_init)):
            classes[i_class + i*len(classes_init)] += chr(97+i)

class_naming_expansion(classes, len(image_load_dir))

for i, i_class in enumerate(classes):
    target_class_path = os.path.join(home_save_dir, target_dir, i_class)
    if not os.path.isdir( target_class_path ):
        print("The specified target class", i_class, "directory does not exist. Please create one.")
    if os.listdir( target_class_path ):
        print("The specified target class", i_class, "directory is not empty. Please correct this.")

imag_list = []
for i_class in classes_init:
    global_list = []
    for i, i_imagedir in enumerate(image_load_dir):
        for ext in extensions:
            if i == 0:
                path = os.path.join( home_load_dir, i_imagedir, i_class+"_lMW_0_bMW_90", '*.' + ext )
                if glob.glob(path): #otherwise the list is empty
                    global_list.extend( glob.glob(path) ) 
                    imag_list.append( global_list )
            elif i == 1:
                path = os.path.join( home_load_dir, i_imagedir, i_class, '*.' + ext )
                if glob.glob(path): #otherwise the list is empty
                    global_list.extend( glob.glob(path) ) 
                    imag_list.append( global_list )

if isListEmpty(imag_list): 
    print("No pictures were selected!")
    quit()

#############################
###PICTURE MANIPULATION######
#############################
assert len(classes) == len(imag_list) # all load folders should have the same number of class folders

for i, iFolder in enumerate(imag_list):
    final_path = os.path.join( home_save_dir, target_dir, classes[i] ) 
    for iPicture in iFolder:
        temp1, temp2 = os.path.splitext( os.path.basename(iPicture) )
        extension_temp = os.path.splitext(iPicture)[1][1:]
        if extension_temp not in extensions:
            print(iPicture, ": that is not a '.jpg' picture (or equivalent)!")
            quit()
        with Image.open(iPicture) as Im:
            if ARGS.type == "contours":
                temp_image = array(Im.convert('L'))
                fig = plt.figure()
                plt.contour(temp_image, origin='image', colors='black', levels=range(20,254,30))
                plt.axis('off')
                if os.path.exists(final_path):
                    fig.savefig( os.path.join(final_path, temp1 + "_contour" + temp2) )  
                else:
                    print("The path of the directory does not exist!")
                    quit()
                plt.close()
            elif ARGS.type == "blur":
                temp_image = Im.filter( ImageFilter.GaussianBlur(radius=float(ARGS.blur_radius)) )
                if os.path.exists(final_path):
                    temp_image.save( os.path.join(final_path, temp1 + "_blur" + temp2) )    
                else:
                    print("The path of the directory does not exist!")
                    quit()
            elif ARGS.type == "blur+contours":
                my_iter_list = [40 if k<4 else 10 for k in range(20)] #this was carefully chosen
                temp_image = Im.filter( ImageFilter.GaussianBlur(radius=float(ARGS.blur_radius)) )
                temp_image = array(temp_image.convert('L'))
                fig = plt.figure()
                plt.contour(temp_image, origin='image', 
                            colors='black', levels=list(var_range(20,254,my_iter_list)))
                plt.axis('off')
                if os.path.exists(final_path):
                    fig.savefig( os.path.join(final_path, temp1 + "_blur+contour" + temp2) )
                else:
                    print("The path of the directory does not exist!")
                    quit()
                plt.close()

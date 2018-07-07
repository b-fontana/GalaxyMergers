import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter
from pylab import *

import os
import glob

import sys
sys.path.append( os.environ['HOME']+'/Code/' )
from MyModule import isListEmpty, var_range

#############################
####PARSING##################
#############################
import argparse
from argparse import RawTextHelpFormatter
parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help='Path to the folder of the images (example: --image_dir=galaxy_photos/). It is assumed that the picture is stored in /data1/alves/. A contour will be drawn around the shapes in the pictures.'
)
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
ARGS, unparsed = parser.parse_known_args()
if (ARGS.image_dir == ""):
    print("Please provide the directory of the images with the '--image_dir' option")
    quit()
if (ARGS.type != "contours" and ARGS.type != "blur" and ARGS.type != "blur+contours"):
    print("Please provide a valid type of picture manipulation thorugh the '--type' option.")
    quit()
if ( (ARGS.type != "blur" and ARGS.type != "blur+contours") and ARGS.blur_radius != "2"):
    print("The blur_radius option can only be used when the chosen type includes blurring. You have selected the", ARGS.type, "type.")
    quit()

################################################
###GET PICTURE ARRAY AND PERFORM CHECKS#########
################################################
home_dir = "/data1/alves/"
target_dir = ARGS.type + ARGS.dir_extension #this directory should already exist
if not os.path.isdir( os.path.join(home_dir, target_dir, ARGS.image_dir) ):
    print("The specified target directory,", os.path.join(home_dir, target_dir, ARGS.image_dir), "does not exist. Please create one.")
    quit()
for _, _, files in os.walk( os.path.join(home_dir, target_dir, ARGS.image_dir), topdown = False ):
    if files:
        print("The specified target directory is not empty. Please correct this.")
        quit()
extensions = [ 'jpg', 'jpeg', 'JPG', 'JPEG']
classes = []
loop_counter = 0
print("[After introducing all classes press \'Enter\' again. Do not write anything in case you want to use the default, which is \'merger\' and \'noninteracting\']")
while True:
    loop_counter += 1
    x = input("Enter one class for which the contour will be drawn: ")
    if x == "":
        if loop_counter == 1:
            classes.extend(('merger', 'noninteracting'))
        break
    classes.append(x)
for i, Cl in enumerate(classes):
    target_class_path = os.path.join(home_dir, target_dir, ARGS.image_dir, Cl)
    if not os.path.isdir( target_class_path ):
        print("The specified target class", Cl, "directory does not exist. Please create one.")
    if os.listdir( target_class_path ):
        print("The specified target class", Cl, "directory is not empty. Please correct this.")

imag_list = []
for Cl in classes:
    global_list = []
    for ext in extensions:
        pathCl = os.path.join( home_dir, ARGS.image_dir, Cl, '*.' + ext )
        if glob.glob(pathCl): global_list.extend( glob.glob(pathCl) ) #otherwise the list is empty
    imag_list.append( global_list  )

if isListEmpty(imag_list): 
    print("No pictures were selected!")
    quit()

#############################
###PICTURE MANIPULATION######
#############################
for i, iFolder in enumerate(imag_list):
    final_path = os.path.join( home_dir, target_dir, ARGS.image_dir, classes[i] ) 
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

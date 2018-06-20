import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
from pylab import *

import os
import glob

import sys
sys.path.append( os.environ['HOME']+'/Code/' )
from MyModule import isListEmpty

#############################
####PARSING##################
#############################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help='Path to the folder of the images (example: --image_dir=galaxy_photos/). It is assumed that the picture is stored in /data1/alves/. A contour will be drawn around the shapes in the pictures.'
)
ARGS, unparsed = parser.parse_known_args()
if (ARGS.image_dir == ""):
    print("Please provide the directory of the images with the '--image_dir' option")
    quit()

#############################
###GET PICTURE ARRAY#########
#############################
target_dir = "contours"
extensions = [ 'jpg', 'jpeg', 'JPG', 'JPEG']
classes = []
loop_counter = 0
while True:
    loop_counter += 1
    x = input("Enter one class for which the contour will be drawn (do not write anything in case you want to use the default): ")
    if x == "":
        if loop_counter == 1:
            classes.extend(('merger', 'noninteracting'))
        break
    classes.append(x)

imag_list = []
for Cl in classes:
    global_list = []
    for ext in extensions:
        pathCl = os.path.join( "/data1/alves/", ARGS.image_dir, Cl, '*.' + ext )
        if glob.glob(pathCl): global_list.extend( glob.glob(pathCl) ) #otherwise the list is empty
    imag_list.append( global_list  )

if isListEmpty(imag_list): 
    print("No pictures were selected!")
    quit()

#############################
####CONTOUR##################
#############################
for i, iFolder in enumerate(imag_list):
    for iPicture in iFolder:
        extension_temp = os.path.splitext(iPicture)[1][1:]
        if extension_temp not in extensions:
            print(iPicture, ": that is not a '.jpg' picture!")
            quit()
        with Image.open(iPicture) as Im:
            temp_image = array(Im.convert('L'))
            fig = plt.figure()
            plt.contour(temp_image, origin='image', colors='black', levels=range(20,254,30))
            plt.axis('off')
            temp1, temp2 = os.path.splitext( os.path.basename(iPicture) )
            final_path = os.path.join( "/data1/alves", target_dir, ARGS.image_dir, classes[i] ) 
            if os.path.exists(final_path):
                fig.savefig( os.path.join(final_path,temp1 + "_contour" + temp2) )  
            else:
                print("The path of the directory does not exist!")
                quit()
            plt.close()


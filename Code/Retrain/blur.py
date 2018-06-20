import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter
from pylab import *

import os
import glob

############################# 
####PARSING##################
#############################                                                                            
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help='Path to the folder of the images (example: --image_dir=galaxy_photos/). It is assumed that the picture is stored in /data1/alves/. The pictures will be blured.'
)
ARGS, unparsed = parser.parse_known_args()
if (ARGS.image_dir == ""):
    print("Please provide the directory of the images with the '--image_dir' option")
    quit()

#############################
####BLUR#####################
#############################
home="/data1/alves/"
target_dir="blur/"
extension=".jpg"
classes=["before/","during/","after/"]
strings=[home+ARGS.image_dir+classes[0],
         home+ARGS.image_dir+classes[1],
         home+ARGS.image_dir+classes[2]]
imag_list = [glob.glob(strings[0]+"*jpg"),
             glob.glob(strings[1]+"*jpg"),
             glob.glob(strings[2]+"*jpg")]

for i,iFolder in enumerate(imag_list):
    for iPicture in iFolder:
        with Image.open(iPicture) as Im:
            temp = Im.filter(ImageFilter.GaussianBlur(radius=2))
            temp.save(home + target_dir + ARGS.image_dir + classes[i] +
                        os.path.splitext( os.path.basename(iPicture) )[0] + "_blur.jpg")

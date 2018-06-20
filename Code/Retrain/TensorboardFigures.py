import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

########################
###PARSING##############
######################## 
parser = argparse.ArgumentParser()
parser.add_argument(
  '--filename',
  type=str,
  default='',
  help='Name of the file that contains the data.'
)
ARGS, unparsed = parser.parse_known_args()
if(ARGS.filename==""): 
  print("Please provide the name of the file where the data is stored.")
  quit()

########################
###STORING##############
######################## 
#myfile = open("/home/alves/run_validation-tag-cross_entropy_1.csv",'r')
myfile = open(ARGS.filename,'r')

values=[]
time_steps=[]
myfile.readline() #skip first line
while True:
      line = myfile.readline()
      if line == "": #eof
        break 
      values.append( float(line.strip().split(',')[1]) )
      time_steps.append( float(line.strip().split(',')[2]) ) 
########################
###PLOTTING#############
########################
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(values,time_steps,color='green')
ax.set(xlabel='Steps', ylabel='Loss function')
ax.grid()
fig.savefig("LossFunction.png")

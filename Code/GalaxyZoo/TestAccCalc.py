#!/usr/bin/env python3

with open('/data1/alves/tf_files/bad_contours/classification_results.txt', 'r') as myfile:
  counter=0
  counter_total=0
  number_of_elements = len(myfile.readline().split())
  myfile.seek(0)
  values = [[] for i in range(number_of_elements)]

  while True:
    line = myfile.readline()
    if line == "":
      break
    for iElem in range(number_of_elements):
      values[iElem] = float(line.split()[0])
      value2 = float(line.split()[1])
    #print("%f\t%f" % (value1,value2))
    if value1 > value2:
      counter += 1
      counter_total += 1
  
  print("Test accuracy: ", (counter/counter_total)*100, "%.")
        




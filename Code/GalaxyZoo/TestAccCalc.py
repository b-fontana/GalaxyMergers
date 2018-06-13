myfile = open("Data/GalaxyZoo/tf_model_files/classification_results.txt",'r')
counter=0
counter_total=0
value1=0.
value2=0.

while True:
  line = myfile.readline()
  if line == "":
    break
  value1 = float(line.split()[0])
  value2 = float(line.split()[1])
  print("%f\t%f" % (value1,value2))
  if value1 > value2:
    counter += 1
  counter_total += 1

print("Test accuracy: ", (counter/counter_total)*100, "%.")





import matplotlib
import matplotlib.pyplot as plt
import numpy as np

f_train = open("/home/alves/test.txt")
split = [l.split('\t') for l in f_train.readlines()]
loss_train = [float(split[k][0]) for k in range(len(split))]
acc_train = [float(split[k][1]) for k in range(len(split))]
loss_valid = [float(split[k][2]) for k in range(len(split))]
acc_valid = [float(split[k][3]) for k in range(len(split))]

fig, ax = plt.subplots()
ax.plot( np.array(range(len(loss_train)))+1, np.array(loss_train), label='Training' )
ax.plot( np.array(range(len(loss_valid)))+1, np.array(loss_valid), label='Validation' )
ax.legend()
plt.xticks(range(0,len(loss_train)+1,25))
ax.set(xlabel='Epochs', ylabel='Binary crossentropy loss')
fig.savefig("loss.png")


fig2, ax2 = plt.subplots()
ax2.plot( np.array(range(len(acc_train)))+1, np.array(acc_train), label='Training' )
ax2.plot( np.array(range(len(acc_valid)))+1, np.array(acc_valid), label='Validation' )
ax2.legend()
plt.xticks(range(0,len(acc_train)+1,25))
ax2.set(xlabel='Epochs', ylabel='Binary crossentropy accuracy')
fig2.savefig("accuracy.png")

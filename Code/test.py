import tensorflow as tf
from tensorflow.python.framework import ops

scalar1 = tf.Variable(3, tf.int32)
matrix1 = tf.Variable([[3,4],[6,7],[7,8]], tf.int32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

print(scalar1.dtype)
print(matrix1.dtype)

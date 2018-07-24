import tensorflow as tf
import numpy as np

class Picture:
    def __init__(self, name=""):
        self.name = name

    def tf_decoder(self, dims):
        """
        Graph that decodes a jpeg image.
        1. The graph
        2. The tensor which provides the decoded picture in its final form
        3. The placeholder for the picture name                                                                 """
        g = tf.Graph()
        with g.as_default():
            picture_name_tensor = tf.placeholder(tf.string)
            picture_contents = tf.read_file(picture_name_tensor)
            picture = tf.image.decode_jpeg(picture_contents, dct_method="INTEGER_ACCURATE")
            picture_as_float = tf.image.convert_image_dtype(picture, tf.float32)
            picture_4d = tf.expand_dims(picture_as_float, 0)
            resize_shape = tf.stack([dims[0], dims[1]])
            resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
            final_tensor =  tf.image.resize_bilinear(picture_4d, resize_shape_as_int,
                                                     align_corners=True)*255
            #if grey-scale is activated, consider only one column
            if dims[2]==1: 
                final_tensor = tf.slice(final_tensor, [0,0,0,0], [-1,-1,-1,1])
        return g, picture_name_tensor, final_tensor
            
    def np_decoder(self, picture_name, height, width):
        """
        Converts a picture to an array using numpy.
        Equivalent to the 'tf_decoder' function, but much slower.
        """
        image = Image.open(picture_name)
        image = image.resize((height,width), Image.LANCZOS)
        image = np.array(image, dtype=np.int32) #the number can be at most 255
        return np.expand_dims(image, axis=0)

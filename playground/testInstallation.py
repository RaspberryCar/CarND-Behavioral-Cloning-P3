import tensorflow as tf
import cv2

print("")
print("OpenCV     version: {}".format(cv2.__version__))
print("Keras      version: {}".format(tf.keras.__version__))
print("Tensorflow version: {}".format(tf.__version__))
print("Num Physical  GPUs: {}".format(len(tf.config.list_physical_devices('GPU'))))
print("Num Logical   GPUs: {}".format(len(tf.config.list_logical_devices('GPU'))))

# https://www.tensorflow.org/guide/gpu#logging_device_placement
tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)


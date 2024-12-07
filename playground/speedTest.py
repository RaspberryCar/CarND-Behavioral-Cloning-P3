import platform

import tensorflow as tf
# from tensorflow.python.compiler.mlcompute import mlcompute
# from tensorflow.python.framework.ops import disable_eager_execution
import datetime

t_set = lambda: datetime.datetime.now().astimezone().replace(microsecond=0)
t_diff = lambda t: str(datetime.datetime.now().astimezone().replace(microsecond=0) - t)
t_stamp = lambda t=None: str(t) if t else str(t_set())

tStart = t_set()

# disable_eager_execution()
# mlcompute.set_mlc_device(device_name='gpu')  # Available options are 'cpu', 'gpu', and 'any'.
tf.config.run_functions_eagerly(False)

print("Python     version: {}".format(platform.python_version()))
print("TensorFlow version: {}".format(tf.__version__))
# print("Keras      version: {}".format(tf.keras.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
# print("Cuda version: {}".format(tf_build_info.cuda_version_number))
# print("Cudnn version: {}".format(tf_build_info.cudnn_version_number))
print("Num Physical GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num Logical  GPUs Available: ", len(tf.config.list_logical_devices('GPU')))

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

print("")
print("Time complete elapsed: {}".format(t_diff(tStart)))

import platform

import tensorflow as tf
import datetime

t_set = lambda: datetime.datetime.now().astimezone().replace(microsecond=0)
t_diff = lambda t: str(datetime.datetime.now().astimezone().replace(microsecond=0) - t)
t_stamp = lambda t=None: str(t) if t else str(t_set())

tStart = t_set()

tf.config.run_functions_eagerly(False)

print("Python     version: {}".format(platform.python_version()))
print("TensorFlow version: {}".format(tf.__version__))
# print("Keras      version: {}".format(tf.keras.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
# print("Cuda version: {}".format(tf_build_info.cuda_version_number))
# print("Cudnn version: {}".format(tf_build_info.cudnn_version_number))
print("Num Physical GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num Logical  GPUs Available: ", len(tf.config.list_logical_devices('GPU')))

print("")
print("An other test from https://medium.com/bluetuple-ai/how-to-enable-gpu-support-for-tensorflow-or-pytorch-on-macos-4aaaad057e74")
tStart = t_set()
cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()
model = tf.keras.applications.ResNet50(
  include_top=True,
  weights=None,
  input_shape=(32, 32, 3),
  classes=100,)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=4096)
print("Time complete elapsed: {}".format(t_diff(tStart)))

import platform
import plotext as plt
import tensorflow as tf
import datetime
from time import sleep, perf_counter as pc

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
cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100, )

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])


class TestRun():  # leave this empty
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.time_str = 0
        self.time = 0


testArray = [
    # TestRun(32),
    TestRun(64),
    TestRun(128),
    TestRun(256),
    TestRun(512),
    TestRun(1024),
    TestRun(2048),
    TestRun(4096),
    TestRun(8192),
    TestRun(16384),
    TestRun(32768),
    TestRun(65536),
]

for batch_run in testArray:
    print("batchSize=" + str(batch_run.batch_size))
    tStart = t_set()
    t0 = pc()
    model.fit(x_train, y_train, epochs=5, batch_size=batch_run.batch_size)
    diff_time = t_diff(tStart)
    batch_run.time_str = diff_time
    batch_run.time = int(pc() - t0)
    print("Time complete elapsed: {} {} s".format(batch_run.time_str, batch_run.time))

for batch_run in testArray:
    print("Time complete elapsed: {} {} for batchsize={}".format(batch_run.time_str, batch_run.time, batch_run.batch_size))


def batchList(val):
    return val.batch_size


def timeList(val):
    return val.time


batch_list = list(map(batchList, testArray))
time_list = list(map(timeList, testArray))

plt.bar(batch_list, time_list)
plt.title(platform.uname().node + " " + platform.uname().machine)
plt.xlabel("batch size")
plt.ylabel("Time")
plt.show()

# https://medium.com/analytics-vidhya/m1-mac-mini-scores-higher-than-my-nvidia-rtx-2080ti-in-tensorflow-speed-test-9f3db2b02d74

from time import perf_counter

from time import sleep, perf_counter as pc
import platform
import tensorflow as tf
import plotext as plt

# download cifar10 dataset
cifar10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_set_count = len(train_labels)
test_set_count = len(test_labels)


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
    # TestRun(1024),
    # TestRun(2048),
    # TestRun(4096),
    # TestRun(8192),
    # TestRun(16384),
    # TestRun(32768),
    # TestRun(65536),
]

# normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# create ML model using tensorflow provided ResNet50 model, note the [32, 32, 3] shape because that's the shape of cifar
model = tf.keras.applications.ResNet50(
    include_top=True, weights=None, input_tensor=None,
    input_shape=(32, 32, 3), pooling=None, classes=10
)

# CIFAR 10 labels have one integer for each image (between 0 and 10)
# We want to perform a cross entropy which requires a one hot encoded version e.g: [0.0, 0.0, 1.0, 0.0, 0.0...]
train_labels = tf.one_hot(train_labels.reshape(-1), depth=10, axis=-1)

# Do the same thing for the test labels
test_labels = tf.one_hot(test_labels.reshape(-1), depth=10, axis=-1)

for batch_run in testArray:
    print("batchSize=" + str(batch_run.batch_size))
    # setup start time
    t1_start = perf_counter()
    t0 = pc()

    # compile ML model, use non sparse version here because there is no sparse data.
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    # train ML model
    model.fit(train_images, train_labels, epochs=5, batch_size=batch_run.batch_size)

    # evaluate ML model on test set
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    # setup stop time
    t1_stop = perf_counter()
    batch_run.time = int(t1_stop - t1_start)
    # print results
    print('\n')
    print(f'Training set contained {train_set_count} images')
    print(f'Testing set contained {test_set_count} images')
    print(f'Model achieved {test_acc:.2f} testing accuracy')
    print(f'Training and testing took {batch_run.time :.2f} seconds for batch_size={batch_run.batch_size}')


def batchList(val):
    return val.batch_size


def timeList(val):
    return val.time


print("GPU device name:")
print(tf.test.gpu_device_name())
batch_list = list(map(batchList, testArray))
time_list = list(map(timeList, testArray))

plt.plot(batch_list, time_list)
plt.title(__file__.rsplit("/", 1)[1].split('.')[0] + ".py " + platform.uname().node + " " + platform.uname().machine)
plt.xlabel("batch size")
plt.ylabel("Time")
plt.show()

for batch_run in testArray:
    print("Time complete elapsed: {} {} for batchsize={}".format(batch_run.time_str, batch_run.time, batch_run.batch_size))

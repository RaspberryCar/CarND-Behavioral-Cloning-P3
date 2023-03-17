import csv
import cv2
import numpy as np
import tensorflow as tf
import datetime

t_set = lambda: datetime.datetime.now().astimezone().replace(microsecond=0)
t_diff = lambda t: str(datetime.datetime.now().astimezone().replace(microsecond=0) - t)
t_stamp = lambda t=None: str(t) if t else str(t_set())

lines = []
tStart = t_set()

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# with open('./data_track2/driving_log.csv') as csvfile:
#  reader = csv.reader(csvfile)
#  for line in reader:
#    lines.append(line)

images = []
measurements = []

print("Getting data....{} lines".format(len(lines)))
correction = 0.2
for line in lines:
    for i in range(3):
        source_path = line[i]
        print("source|" + str(line))
        filename = source_path.split('/')[-1]
        current_path = filename.strip()
        measurement = float(line[3])
        print("path is|" + current_path + " measurement=" + str(measurement))
        image = cv2.imread(current_path)
        # print(image)
        image_rgb = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        images.append(image_rgb)

        if i == 1:
            measurement = measurement + correction

        if i == 2:
            measurement = measurement - correction

        measurements.append(measurement)

augmented_images, augmented_measurements = [], []

t = t_set()

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from tensorflow.keras.layers import Conv2D

# Nvidia's CNN architecture
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=X_train[0].shape))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Dropout(0.2))
model.add(Conv2D(24, (5, 5), activation="relu"))
model.add(Conv2D(36, (5, 5), activation="relu"))
model.add(Conv2D(48, (5, 5), activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# model = Sequential()
# model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=(160, 320, 3)))

# model.add(Cropping2D(cropping=((80, 25), (0, 0))))

# model.add(Convolution2D(6, 5, 5, activation="relu"))
# model.add(MaxPooling2D())

# model.add(Convolution2D(6, 5, 5, activation="relu"))
# model.add(MaxPooling2D())

# use dropout to eliminate overfitting and improve the validation loss
# model.add(Dropout(.3))

# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
print("model.summary ", model.summary())
print("")

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=4)

print("")
modelName = 'model.tflite'

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TF Lite model
with tf.io.gfile.GFile(modelName, 'wb') as f:
    f.write(tflite_model)

print("")
print("Time ML       elapsed: {}".format(t_diff(t)))
print("Time complete elapsed: {}".format(t_diff(tStart)))

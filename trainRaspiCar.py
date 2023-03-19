import csv
import glob
import random
from configparser import ConfigParser

import numpy as np
import tensorflow as tf
import cv2
from keras.layers import Conv2D
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential

import datetime

t_set = lambda: datetime.datetime.now().astimezone().replace(microsecond=0)
t_diff = lambda t: str(datetime.datetime.now().astimezone().replace(microsecond=0) - t)
t_stamp = lambda t=None: str(t) if t else str(t_set())

tStart = t_set()


class TData:
    def __init__(self, config_filepath):
        self._init()
        self.configure(config_filepath)

    def configure(self, config_filepath):
        parser = ConfigParser()
        parser.read(config_filepath)
        self._csv_path = parser.get('general', 'csv_path')
        self._img_path = parser.get('general', 'img_path')
        self._model_name = parser.get('general', 'model_name') + datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
        self._image_width = int(parser.get('general', 'image_width'))
        self._image_height = int(parser.get('general', 'image_heigth'))
        self._random_count = int(parser.get('general', 'random_count'))
        self._angle_max = float(parser.get('general', 'angle_max'))
        self._angle_min = float(parser.get('general', 'angle_min'))
        self._correction = float(parser.get('general', 'correction'))
        self._random_mode = True if parser.get('general', 'random_mode') == "True" else False
        print("configure done")

    def _init(self):
        self._lines = []
        self._images = []
        self._measurements = []
        self._augmentated_images = []
        self._augmentated_measurements = []

    def _fetch_lines(self, filename):
        self._lines = []
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                self._lines.append(line)

    def fetch(self):
        if self._random_mode == True:
            self._fetch_random()
        else:
            self._fetch_from_path()

    def _fetch_from_path(self):
        self._init()
        files = glob.glob(self._img_path + "*.jpg")
        print("files=" + str(len(files)) + " " + self._img_path)
        for f in files:
            try:
                line_token_list = f.split('_')
                # carData1678530861448_0_STOP_0.jpg
                # PREFIX-timestamp_<STEERING>_<DIRECTION>_<SPEED>.jpp
                steering = line_token_list[1]
                direction = line_token_list[2]
                speed = line_token_list[3].split(".")[0]  # without <.jpg>
                normalize = (float(steering) - self._angle_min) / (self._angle_max - self._angle_min)
                print("files={} steering={}/{} direction={} speed={}".format(f, steering, normalize, direction, speed))
                # maybe convert to radian with from -pi/4 to pi/4
                # and add correction required
                # old format:
                # center_2023_03_14_14_56_39_822.jpg
                # 1. filename must be in format xxx_<ANGLE_DAT>_**.jpg

                measurement = 2 * normalize - 1
                image = cv2.imread(f)
                image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
                image_rgb = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            except Exception as e:
                print(e)
            else:
                self._images.append(image_rgb)
                self._measurements.append(measurement)
        self._add_augmenged_information()

    def _fetch_real(self):
        self._init()
        self._fetch_lines(self._csv_path)
        for line in self._lines:
            for i in range(3):
                source_path = line[i]
                filename = source_path.split('/')[-1]
                current_path = self._img_path + filename
                try:
                    measurement = float(line[3])
                    image = cv2.imread(current_path)
                    image_rgb = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                except Exception as e:
                    print(e)
                else:
                    if (i == 1): measurement = measurement + self._correction
                    if (i == 2): measurement = measurement - self._correction
                    self._images.append(image_rgb)
                    self._measurements.append(measurement)
        self._add_augmenged_information()

    def _add_augmenged_information(self):
        for image, measurement in zip(self._images, self._measurements):
            self._augmentated_images.append(image)
            self._augmentated_measurements.append(measurement)
            self._augmentated_images.append(cv2.flip(image, 1))
            self._augmentated_measurements.append(measurement * -1)

    def train(self):
        X_train = np.array(self._augmentated_images)
        y_train = np.array(self._augmentated_measurements)
        model = Sequential()
        model.add(Cropping2D(cropping=((10, 2), (0, 0)), input_shape=X_train[0].shape))
        model.add(Lambda(lambda x: (x / 255.0) - 0.5))
        model.add(Dropout(0.2))
        model.add(Conv2D(24, (5, 5), activation="relu", padding='same'))
        model.add(MaxPooling2D())
        model.add(Conv2D(36, (5, 5), activation="relu", padding='same'))
        model.add(MaxPooling2D())
        model.add(Conv2D(48, (5, 5), activation="relu", padding='same'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation="relu", padding='same'))
        model.add(MaxPooling2D())
        model.add(Conv2D(64, (3, 3), activation="relu", padding='same'))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(1164))
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=4)
        model.save(self._model_name + ".h5")
        self._generate_tflite()
        results = model.evaluate(X_train, y_train, verbose=0)
        print("Evaluate:{}".format(results))

    def print_info(self):
        print("Info (augmented measurements): {}".format(len(self._augmentated_measurements)))

    def _generate_random_image(self, width=128, height=128):
        rand_pixels = np.random.randint(255, size=(height, width, 3), dtype=np.uint8)
        # cv2.imshow('RGB',rand_pixels)
        # cv2.waitKey(0)
        return rand_pixels

    def _generate_random_angle(self):
        x = random.uniform(self._angle_min, self._angle_max)
        normalize = (x - self._angle_min) / (self._angle_max - self._angle_min)
        return 2 * normalize - 1

    def _fetch_random(self):
        self._init()
        for _ in range(self._random_count):
            image = self._generate_random_image(self._image_width, self._image_height)
            measurement = self._generate_random_angle()
            self._images.append(image)
            self._measurements.append(measurement)
        self._add_augmenged_information()

    def _generate_tflite(self):
        model = tf.keras.models.load_model(self._model_name + ".h5")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        open(self._model_name + ".tflite", "wb").write(tflite_model)
        print("TFFile:", self._model_name + ".tflite")


if __name__ == "__main__":
    try:
        d = TData("./config.ini")
        d.fetch()
        d.print_info()
        d.train()
        print("")
        print("Time complete elapsed: {}".format(t_diff(tStart)))
    except Exception as e:
        print(e)

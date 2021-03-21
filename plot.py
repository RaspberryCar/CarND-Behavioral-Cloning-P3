import base64
import datetime
import glob
import os
from configparser import ConfigParser

import matplotlib.pyplot as plt
import mpld3
import pandas as pd
from mpld3 import plugins

# Define some CSS to control our custom labels
css = """
img {
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 5px;
  width: 150px;
}

table {
  background-color: #f1f1c1;
}
"""


class TPlot:
    def __init__(self, config_filepath):
        self._init()
        self.configure(config_filepath)

    def configure(self, config_filepath):
        parser = ConfigParser()
        parser.read(config_filepath)
        self._csv_path = parser.get('general', 'csv_path')
        self._img_path = parser.get('general', 'img_path')
        self._simu_img_path = parser.get('general', 'simu_img_path')
        self._model_name = parser.get('general', 'model_name')
        self._image_width = int(parser.get('general', 'image_width'))
        self._image_height = int(parser.get('general', 'image_heigth'))
        self._random_count = int(parser.get('general', 'random_count'))
        self._angle_max = float(parser.get('general', 'angle_max'))
        self._angle_min = float(parser.get('general', 'angle_min'))
        self._correction = float(parser.get('general', 'correction'))
        self._random_mode = True if parser.get('general', 'random_mode') == "True" else False

    def _init(self):
        self._lines = []
        self._images = []
        self._measurements = []
        self._augmentated_images = []
        self._augmentated_measurements = []
        self._df_real = pd.DataFrame()
        self._df_simu = pd.DataFrame()

    def fetch(self):
        self._fetch_from_path()
        # self._fetch_from_csv()

    def normalize(self, angle):
        x = (angle - self._angle_min) / (self._angle_max - self._angle_min)
        return 2 * x - 1

    def get_angle(self, path):
        return float(os.path.basename(path).split('_')[1])

    def get_time(self, path):
        return int(os.path.basename(path).split('_')[0])

    def _fetch_from_path(self):
        files = glob.glob(self._img_path + "*.jpg")
        self._df_real['file'] = files
        self._df_real['base64'] = self._df_real.apply(lambda row: self.get_base64_encoded_image(row.file), axis=1)
        self._df_real['time'] = self._df_real.apply(lambda row: self.get_time(row.file), axis=1)
        self._df_real['time'] = pd.to_datetime(self._df_real['time'], unit='s')
        self._df_real['angle_degree'] = self._df_real.apply(lambda row: self.get_angle(row.file), axis=1)
        self._df_real['angle_normalize'] = self._df_real.apply(lambda row: self.normalize(row.angle_degree), axis=1)

    def get_base64_encoded_image(self, image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def get_simupath(self, image_path):
        filename = image_path.split('/')[-1]
        return self._simu_img_path + filename

    def get_time_image(self, image_path):
        time = image_path.split('_')
        return datetime.datetime(year=int(time[1]), month=int(time[2]), day=int(time[3]),
                                 hour=int(time[4]), minute=int(time[5]), second=int(time[6]),
                                 microsecond=int(time[7].split(".")[0]))

    def _fetch_from_csv(self):
        self._df_simu = pd.read_csv(self._csv_path, usecols=[0, 1, 2, 3],
                                    names=['cfile', 'lfile', 'rfile', 'angle_normalize'])
        self._df_simu['cfile'] = self._df_simu.apply(lambda row: self.get_simupath(row.cfile), axis=1)
        self._df_simu['cbase64'] = self._df_simu.apply(lambda row: self.get_base64_encoded_image(row.cfile), axis=1)
        self._df_simu['ctime'] = self._df_simu.apply(lambda row: self.get_time_image(row.cfile), axis=1)

        # self._df_simu['lfile'] = self._df_simu.apply(lambda row: self.get_simupath(row.lfile), axis=1)
        # self._df_simu['lbase64'] = self._df_simu.apply(lambda row: self.get_base64_encoded_image(row.lfile), axis=1)

        # self._df_simu['rfile'] = self._df_simu.apply(lambda row: self.get_simupath(row.rfile), axis=1)
        # self._df_simu['rbase64'] = self._df_simu.apply(lambda row: self.get_base64_encoded_image(row.rfile), axis=1)
        # 3177
        # 3443
        print(self._df_simu.angle_normalize.size)

    def prepare_plot_simu(self, fig, ax):
        ax.grid(True, alpha=0.3)
        N = self._df_simu.angle_normalize.size
        labels = []
        for i in range(N):
            label = self._df_simu.iloc[[i], :]
            src = str('data:image/jpeg;base64,' + label.cbase64.values[0])
            html = '<img src="' + src + '" width="' + str(self._image_width / 2) + '" height="' + str(
                self._image_height / 2) + '">'
            html += '<table>'
            html += '<tr><td><b>Row:</b></td><td>' + str(i) + '</td></tr>'
            html += '<tr><td><b>Path:</b></td><td>' + str(label.cfile.values[0]) + '</td></tr>'
            # html += '<tr><td><b>Angle(°):</b></td><td>'+ str(label.angle_degree.values[0]) +'</td></tr>'
            html += '<tr><td><b>Normalized angle:</b></td><td>' + str(label.angle_normalize.values[0]) + '</td></tr>'
            html += '<tr><td><b>Unix timestamp:</b></td><td>' + str(label.ctime.values[0]) + '</td></tr>'
            html += '</table>'
            labels.append(html)
        points = ax.plot(self._df_simu.ctime, self._df_simu.angle_normalize, 'o', color='b', mec='k', ms=15, mew=1,
                         alpha=.6)

        ax.set_xlabel('angle_normalize')
        ax.set_ylabel('time')
        tooltip = plugins.PointHTMLTooltip(points[0], labels, voffset=10, hoffset=10, css=css)
        plugins.connect(fig, tooltip)

    def prepare_plot_real(self, fig, ax):
        ax.grid(True, alpha=0.3)
        N = self._df_real.angle_degree.size
        labels = []
        for i in range(N):
            label = self._df_real.iloc[[i], :]
            src = str('data:image/jpeg;base64,' + label.base64.values[0])
            html = '<img src="' + src + '" width="' + str(self._image_width / 2) + '" height="' + str(
                self._image_height / 2) + '">'
            html += '<table>'
            html += '<tr><td><b>Row:</b></td><td>' + str(i) + '</td></tr>'
            html += '<tr><td><b>Path:</b></td><td>' + str(label.file.values[0]) + '</td></tr>'
            html += '<tr><td><b>Angle(°):</b></td><td>' + str(label.angle_degree.values[0]) + '</td></tr>'
            html += '<tr><td><b>Normalized angle:</b></td><td>' + str(label.angle_normalize.values[0]) + '</td></tr>'
            html += '<tr><td><b>Unix timestamp:</b></td><td>' + str(label.time.values[0]) + '</td></tr>'
            html += '</table>'
            labels.append(html)
        points = ax.plot(self._df_real.time, self._df_real.angle_normalize, 'o', color='b', mec='k', ms=15, mew=1,
                         alpha=.6)

        ax.set_xlabel('angle_normalize')
        ax.set_ylabel('time')
        tooltip = plugins.PointHTMLTooltip(points[0], labels, voffset=10, hoffset=10, css=css)
        plugins.connect(fig, tooltip)

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.1, wspace=0.1)
        fig.suptitle('Steering angle')
        self.prepare_plot_real(fig, ax)
        # self.prepare_plot_simu(fig, ax[1])
        mpld3.show()

    def show(self):
        self.plot()
        # print(self._df)


if __name__ == "__main__":
    try:
        d = TPlot("./config.ini")
        d.fetch()
        d.show()
    except Exception as e:
        print("error:", e)

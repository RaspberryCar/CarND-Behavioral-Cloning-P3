import platform
import plotext as plt


class TestRun():  # leave this empty
    def __init__(self, batch_size, time):
        self.batch_size = batch_size
        self.time = time


print(platform.uname())
testArray = [
    # TestRun(32, 1613),
    # TestRun(64, 803),
    # TestRun(128, 399),
    TestRun(256, 208),
    TestRun(512, 128),
    TestRun(1024, 96),
    TestRun(2048, 88),
    TestRun(4096, 84),
    TestRun(8192, 84),
    TestRun(16384, 93),
    TestRun(32768, 110),
    TestRun(65536, 142),
]


def batchList(val):
    return val.batch_size


def timeList(val):
    return val.time


batch_list = list(map(batchList, testArray))
time_list = list(map(timeList, testArray))

# plt.hist(data1, 60, label = "mean 0")
# plt.bar(batch_list, time_list)
plt.plot(batch_list, time_list)
plt.title(platform.uname().node + " " + platform.uname().machine)
plt.xlabel("batch size")
plt.ylabel("Time")
plt.show()

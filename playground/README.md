##  Speedtest

`python playground/speedTest.py`

### Tensorflow 2.4.0

|  Jetson nano   | Apple M1 | Apple M1 pro |
|-----|----------|--------------|
|   0:02:14  | 0:00:33  | 0:00:29      |

### Tensorflow 2.9 with tensorflow-metal

https://developer.apple.com/metal/tensorflow-plugin/

`speedtest.py` removed `mlcompute`
```
pip install --force-reinstall -v tensorflow-macos==2.9
pip install --force-reinstall -v tensorflow-metal==0.5.0
```

| M1  |    M1 pro   |
|-----|-----|
| 0:00:42 |   0:00:48  |

##  Speedtest BIG

`python playground/speedTestBig.py`

### tensorflow 2.4.0 local

| M1              |    M1 pro   |
|-----------------|-----|
| 2263.53 seconds |   1751.69 seconds  |

### tensorflow 2.9 CI

| M1  |    M1 pro   |
|-----|-----|
| 1320.85 seconds |   1103.40 seconds |
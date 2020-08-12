# dnntune
A DNN tuning framework for mobile devices
## Overview

This project is built for auto-benchmarking AI workloads on android mobile phones.
## Pre-requirments and Setup

1. One android phone with system version>=5.0 and 64-bit arm processor
2. Make sure the Android devolopment mode is enabled
3. Install android SDK tools on your PC and can connect to the mobile phone via adb
4. For performance profiling, it would be better to root the mobile phone
5. Download [Snapdragon Profiler](https://developer.qualcomm.com/software/snapdragon-profiler) and install it (optional for energy profiling)

## Download models and benchmark model tools
All the models and benchmark model tools are hosted on the google drive.
1. Clike the [link](https://drive.google.com/drive/folders/1oJQbj3X-raU6WWeqwoZZk3DKYsB9r145?usp=sharing)
and download `tf_benchmark_model` and `tflite_benchmark_model`.


```shell
adb push path/to/tf_benchmark_model /data/local/tmp/
adb shell "chmod +x /data/local/tmp/tf_benchmark_model"
adb push path/to/tflite_benchmark_model /data/local/tmp/
adb shell "chmod +x /data/local/tmp/tflite_benchmark_model"
```

The model file who's name is start with 'frozen_' is TensorFlow model.
The model file who's name is end with '.tflite' is TFLite model.
The model file who's name is end with 'quantized.tflite' is TFLite quantized model.
Download the model you are interested in.

2. Push model(s) to your android devices.

```shell
mkdir /sdcard/dnntune_models
adb push path/to/your_downloaded_model_file /sdcard/dnntune_models
```

3. Start benchmarking. Assume you have `mobilenet-v1.tflite` in `/sdcard/dnntune_models`
```shell
python dnntune_models.py --framework TFLite --model_name mobilenet-v1 --device CPU --thread_number 2 --use_quantization 0
```

4. Supported models are listed in `dnntune_models.py`.


## More
Our work is based on the open source frameworks including [Tensorflow](https://github.com/tensorflow/tensorflow),
 [TFLite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite) and [MACE](https://github.com/XiaoMi/mace), [MNN](https://github.com/alibaba/MNN).
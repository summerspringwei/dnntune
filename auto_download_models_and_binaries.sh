cd /tmp/
wget -c https://drive.google.com/drive/folders/1oJQbj3X-raU6WWeqwoZZk3DKYsB9r145?usp=sharing
echo "Download models and benchmark_model executables done."
adb push /tmp/dnntune_models/ /sdcard/
adb shell "cp /sdcard/dnntune_models/tf_benchmark_model /data/local/tmp/"
adb shell "chmod +x /data/local/tmp/tf_benchmark_model"
adb shell "cp /sdcard/dnntune_models/tflite_benchmark_model /data/local/tmp/"
adb shell "chmod +x /data/local/tmp/tflite_benchmark_model"


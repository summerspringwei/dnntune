import argparse
import os

tf_models = ["alexnet", "deeplab-v3-plus-mobilenet-v2", \
    "inception-v1", "inception-v3", "mlp-minist-2048-4096-1024", \
    "minist-rnn-10000", "mobilenet-v1", "mobilenet-v2", \
    "pnasnet-mobile", "resnet-v1-50", "shufflenet-v2", \
    "transformer", "vgg16", "inception-resnet-v2"]
tflite_models = ["deeplab-v3-plus-mobilenet-v2", "deepspeech", \
    "inception-v1", "inception-v3", "mobilenet-v1", "mobilenet-v2", "resnet-v1-50", "vgg16"]

tflite_quantized_models = ["inception-v1", "inception-v3", "mobilenet-v1", "mobilenet-v2", "resnet-v1-50"]

# For tensorflow we need to know the (1)input node names (2) input tenor shapes (3) output node names 
tf_models_map = {"alexnet": ("InputData/X", "1,227,227,3", 'FullyConnected_2/Softmax'),
                 "deeplab-v3-plus-mobilenet-v2": ("sub_7", "1,513,513,3", "ResizeBilinear_2"), 
                 "inception-v1": ("input", "1,224,224,3", "InceptionV1/Logits/Predictions/Reshape_1"),
                 "inception-v3": ("input", "1,299,299,3", "InceptionV3/Predictions/Reshape_1"),
                 "mlp-minist-2048-4096-1024": ("Placeholder", "1,784", "logits/BiasAdd"),
                 "minist-rnn-10000":("images:0", "1,28,28","result_digit:0"),
                 "mobilenet-v1": ("input", "1,224,224,3", "MobilenetV1/Predictions/Reshape_1"),
                 "mobilenet-v2": ("input", "1,224,224,3", "MobilenetV2/Predictions/Reshape_1"),
                 "pnasnet-mobile": ("input", "1,224,224,3", "final_layer/predictions"),
                 "resnet-v1-50": ("input", '1,224,224,3', "resnet_v1_50/predictions/Reshape_1"),
                 "shufflenet-v2": ("input", "1,224,224,3", 'classifier/BiasAdd'),
                 "transformer": ('input_tensor',"1,7", 'model/Transformer/strided_slice_19'),
                 "vgg16": ("input", "1,224,224,3", "vgg_16/fc8/BiasAdd"),
                 "inception-resnet-v2": ("input", "1,224,224,3", 'InceptionResnetV2/Logits/Predictions')
                 }

framework = None # TF, TFLite
model_name = None
device = None # CPU, GPU
thread_number = None
loop_count = None
quantization = None
model_dir_name = "dnntune_models"


def parse_user_args():
    parser = argparse.ArgumentParser(description='Run DNN model benchmark.')
    parser.add_argument('--framework', type=str, required=True,
                        help='The framework used to benchmark ex: [TF, TFLite, MNN]')
    parser.add_argument('--model_name', type=str, required=True,
                        help='The model name ex: [mobilenet-v1, vgg16]')
    parser.add_argument('--device', type=str, required=True,
                        help='The device used to run DNN model ex: [CPU, GPU],\
                             Note TF does not support mobile GPU')
    parser.add_argument('--thread_number', type=int, required=True, default=1,
                        help='The thread number used to run DNN model')
    parser.add_argument('--use_quantization', type=int, required=True,
                        help='Whether run quantized model, ex: [1, 0]')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')

    args = parser.parse_args()
    print(args)
    return args



def get_tf_sh_cmd(args):
    executable = "tf_benchmark_model"
    if args.model_name not in tf_models_map.keys():
        print("Model {} is not supported.".format(args.model_name))
        return None
    (input_layer, input_shape, output_layer) = tf_models_map[args.model_name]
    if args.device == "CPU":
        model_file_path = "/sdcard/{}/frozen_{}.pb".format(model_dir_name, args.model_name)
        sh_cmd = '{}  --graph={} --input_layer={} --input_layer_shape={}  \
            --output_layer={} --input_layer_type="float" --show_run_order=true \
                --show_time=false --max_time=10 --show_flops=true --num_threads={}'.format( \
                    executable, model_file_path, input_layer, input_shape, output_layer, args.thread_number)
        print(sh_cmd)
    else:
        print("Tensorflow does not support running on mobile GPU. Please set device to CPU. exit(0)")
        return None
    return sh_cmd


def get_tflite_sh_cmd(args):
    executable = "tflite_benchmark_model"
    if args.model_name not in tflite_models:
        print("Model {} is not supported.".format(args.model_name))
        return None
    (input_layer, input_shape, output_layer) = tf_models_map[args.model_name]
    use_gpu = 0
    thread_number = args.thread_number
    if args.device == "CPU":
        use_gpu = "false"
    elif args.device == "GPU":
        use_gpu = "true"
        thread_number = 1
    model_file_path = "/sdcard/{}/{}.tflite".format(model_dir_name,args.model_name)

    if args.use_quantization==1:
        if args.model_name not in tflite_quantized_models:
            print("Quantized model {} is not support.".format(args.model_name))
            return None
        model_file_path = "/sdcard/{}/{}-quantized.tflite".format(model_dir_name, args.model_name)
    sh_cmd = '{} --graph={} --input_layer={} --input_layer_shape={} \
        --use_gpu={} --enable_op_profiling=true --num_threads={}'.format( \
            executable, model_file_path,input_layer, input_shape, use_gpu, thread_number)
    print(sh_cmd)
    return sh_cmd

def get_mnn_sh_cmd(args):
    pass


def benchmark_main():
    args = parse_user_args()
    sh_cmd = None
    if args.framework == "TF":
        sh_cmd = get_tf_sh_cmd(args)
    elif args.framework == "TFLite":
        sh_cmd = get_tflite_sh_cmd(args)
    if sh_cmd == None:
        print("Arguments have error!")
    os.system("adb shell /data/local/tmp/{}".format(sh_cmd))

if __name__=="__main__":
    # args = parse_user_args()
    # get_tf_sh_cmd(args)
    # get_tflite_sh_cmd(args)
    benchmark_main()



import os
from data_processing.data_preprocessing import load_wav_file, calculate_mfcc

import tensorflow as tf
import numpy as np
import argparse
import time
import os


def tflite_inference(input_data, tflite_path):
    """Call forwards pass of TFLite file and returns the result.

    Args:
        input_data: Input data to use on forward pass.
        tflite_path: Path to TFLite file to run.

    Returns:
        Output from inference.
    """
    supported_quant_dtypes = (np.int8, np.int16)
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]
    output_dtype = output_details[0]["dtype"]

    # Check if the input/output type is quantized,
    # set scale and zero-point accordingly
    if input_dtype in supported_quant_dtypes:
        input_scale, input_zero_point = input_details[0]["quantization"]
    else:
        input_scale, input_zero_point = 1, 0

    input_data = input_data / input_scale + input_zero_point
    input_data = np.round(input_data) if input_dtype in supported_quant_dtypes else input_data

    if output_dtype in supported_quant_dtypes:
        output_scale, output_zero_point = output_details[0]["quantization"]
    else:
        output_scale, output_zero_point = 1, 0

    interpreter.set_tensor(input_details[0]['index'], tf.cast(input_data, input_dtype))
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)

    return output_data


def load_labels(filename):
    """Read in labels, one label per line."""
    f = open(filename, "r")
    return f.read().splitlines()


def main():
    file_path = "Dataset/testing_list.txt"
    total = 0
    a = 0
    with open(file_path, "r") as f:
        for line in f:
            total = total+1
            line = line[:-1]
            #line = line.strip
            #print(line)
            path1 = "Dataset"  
            split_line = line.split("/")
            label = split_line[0]
            wav_file = split_line[1]
            wav = os.path.join(path1,label)
            wav = os.path.join(wav,wav_file)

            #print(wav)
            
            model_size = os.path.getsize(FLAGS.tflite_path)
            window_size_samples = int(FLAGS.sample_rate * FLAGS.window_size_ms / 1000)
            window_stride_samples = int(FLAGS.sample_rate * FLAGS.window_stride_ms / 1000)
            decoded, sample = load_wav_file(wav, FLAGS.sample_rate)    
            x = calculate_mfcc(decoded, sample, window_size_samples, window_stride_samples, FLAGS.dct_coefficient_count)
            x = tf.reshape(x, [1, -1])
            #start_time = time.time()
            predictions = tflite_inference(x, FLAGS.tflite_path)
            #print(predictions)
            #end_time = time.time()
            #inference_time = end_time - start_time

            # Sort to show labels in order of confidence
            top_k = predictions[0].argsort()[-1:][::-1]

            #print(top_k)
            for node_id in top_k:
                human_string = load_labels(FLAGS.labels)[int(node_id)]
                score = predictions[0,node_id]
                #print("\n")
                #print(f'model predicted:" {human_string}  " with score {score:.5f}')
                #print(f"Inference time: {inference_time:.4f} seconds")
                #print('Model size:', model_size)
                #print(total)
                if human_string == label or human_string == "_unknown_"  or human_string == "_silence_":
                    a = a+1
                    #print("\t")
                    #print(human_string)

                
    accuracy = (a/total)*100
    print(f'model predicted:" {accuracy}" %')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wav', type=str, 
        default='',  
          help='Audio file to be identified.')
    parser.add_argument(
        '--labels', type=str, default='dnn_l/validation_utils/labels.txt',
        help='Path to file containing labels.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs', )
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint', )
    parser.add_argument(
        '--tflite_path',
        type=str,
        default='dnn_small/keras/dnn_s_quantized.tflite',
        help='Path to TFLite file to use for testing.')
    FLAGS, unparsed = parser.parse_known_args()
    main()
# Copyright © 2023 Arm Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from data_processing.data_preprocessing import load_wav_file, calculate_mfcc

import tensorflow as tf
import numpy as np
import argparse
import time
import os


def load_labels(filename):
    """Read in labels, one label per line."""
    f = open(filename, "r")
    return f.read().splitlines()


def main():
    mapp = np.array(['yes','no','up','down','left','right','on','off','stop','go'])
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
            #print("\n")
            #print(wav)        
            #print("\n")
            ok = 0
            for i in mapp:
                #print(i)
                if i == label :
                    ok = 1
                    #print(label)
            if ok == 0:
                label = "_unknown_"  
                #print(label)   
            

            model_size = os.path.getsize(FLAGS.keras_file_path)
            window_size_samples = int(FLAGS.sample_rate * FLAGS.window_size_ms / 1000)
            window_stride_samples = int(FLAGS.sample_rate * FLAGS.window_stride_ms / 1000)
            decoded, sample = load_wav_file(wav, FLAGS.sample_rate)            
            x = calculate_mfcc(decoded, sample, window_size_samples, window_stride_samples, FLAGS.dct_coefficient_count)
            x = tf.reshape(x, [1, -1])

            model = tf.keras.models.load_model(FLAGS.keras_file_path)
            #start_time = time.time()
            predictions = model.predict(x)
            #end_time = time.time()
            #inference_time = end_time - start_time

            # Sort to show labels in order of confidence
            top_k = predictions[0].argsort()[-1:][::-1]
            for node_id in top_k:
                human_string = load_labels(FLAGS.labels)[int(node_id)]
                score = predictions[0,node_id]
                #print(f'model predicted: {human_string} with score {score:.5f}')
                #print(f"Inference time: {inference_time:.4f} seconds")
                #print('Model size:', model_size)
                if human_string == label :  # or human_string == "_unknown_"  or human_string == "_silence_":
                    a = a+1
                    #print(a)
    accuracy = (a/total)*100
    print(accuracy)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wav', type=str, default='',
        help='Audio file to be identified.')
    parser.add_argument(
        '--labels', type=str, default='cnn_small/validation_utils/labels.txt',
        help='Path to file containing labels.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs', )
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=40.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=20.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=10,
        help='How many bins to use for the MFCC fingerprint', )
    parser.add_argument(
        '--keras_file_path',
        type=str,
        default='cnn_small/keras/cnn.h5',
        help='Path to the .h5 Keras model file to use for testing.')
    FLAGS, unparsed = parser.parse_known_args()
    main()

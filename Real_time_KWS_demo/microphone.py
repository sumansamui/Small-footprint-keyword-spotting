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

##########################
#from data_processing.data_preprocessing import load_wav_file, calculate_mfcc
##########################
import tensorflow as tf
import numpy as np
import argparse
import time
import os

import sounddevice as sd
from scipy.io.wavfile import write

#################### data_precessing ##################
from tensorflow.python.ops import gen_audio_ops as audio_ops

def load_wav_file(wav_filename, desired_samples):
    """Loads and then decodes a given 16bit PCM wav file.

    Decoded audio is scaled to the range [-1, 1] and padded or cropped to the desired number of samples.

    Args:
        wav_filename: 16bit PCM wav file to load.
        desired_samples: Number of samples wanted from the audio file.

    Returns:
        Tuple consisting of the decoded audio and sample rate.
    """
    wav_file = tf.io.read_file(wav_filename)
    decoded_wav = audio_ops.decode_wav(wav_file, desired_channels=1, desired_samples=desired_samples)

    return decoded_wav.audio, decoded_wav.sample_rate


def calculate_mfcc(audio_signal, audio_sample_rate, window_size, window_stride, num_mfcc):
    """Returns Mel Frequency Cepstral Coefficients (MFCC) for a given audio signal.

    Args:
        audio_signal: Raw audio signal in range [-1, 1]
        audio_sample_rate: Audio signal sample rate
        window_size: Window size in samples for calculating spectrogram
        window_stride: Window stride in samples for calculating spectrogram
        num_mfcc: The number of MFCC features wanted.

    Returns:
        Calculated mffc features.
    """
    spectrogram = audio_ops.audio_spectrogram(input=audio_signal, window_size=window_size, stride=window_stride,
                                              magnitude_squared=True)

    mfcc_features = audio_ops.mfcc(spectrogram, audio_sample_rate, dct_coefficient_count=num_mfcc)

    return mfcc_features
####################################################

def load_labels(filename):
    """Read in labels, one label per line."""
    f = open(filename, "r")
    return f.read().splitlines()

def record_and_save(output_wav_filename, duration=1, sample_rate=16000):
    print(f"Recording {duration} seconds of audio...")

    # Record audio from the microphone
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()

    # Save the recorded audio to a WAV file
    write(output_wav_filename, sample_rate, audio_data)

    print(f"Audio saved as {output_wav_filename}")

def main():
    # Specify the filename for the saved audio
    output_wav_filename = FLAGS.wav               #"recorded_audio.wav"
    # Set the duration of each recording in seconds
    recording_duration = 2
    model_size = os.path.getsize(FLAGS.keras_file_path)
    window_size_samples = int(FLAGS.sample_rate * FLAGS.window_size_ms / 1000)
    window_stride_samples = int(FLAGS.sample_rate * FLAGS.window_stride_ms / 1000)
    model = tf.keras.models.load_model(FLAGS.keras_file_path)

    while 1:        #True:

        # Record and save audio
        record_and_save(output_wav_filename, duration=recording_duration)
        decoded, sample = load_wav_file(FLAGS.wav, FLAGS.sample_rate)
        x = calculate_mfcc(decoded, sample, window_size_samples, window_stride_samples, FLAGS.dct_coefficient_count)
        x = tf.reshape(x, [1, -1])
        #start_time = time.time()
        predictions = model.predict(x)
        #end_time = time.time()
        #inference_time = end_time - start_time
        # Sort to show labels in order of confidence
        top_k = predictions[0].argsort()[-1:][::-1]
        for node_id in top_k:
            human_string = load_labels(FLAGS.labels)[int(node_id)]
            score = predictions[0,node_id]
            print(f'model predicted: {human_string} with score {score:.5f}')
            #print(f"Inference time: {inference_time:.4f} seconds")
            #print('Model size:', model_size)
        #input("Press Enter to record the next audio or Ctrl+C to exit.")
#############
# _silence_ #
# _unknown_ #
# on        #
# off       #
# stop      #
#############


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wav', type=str, default='microphone.wav',    #'E:/OneDrive - NIT Durgapur/Program/ML_ZOO/ML-zoo-master/models/keyword_spotting/train/micronet_small/New folder/microphone.wav',  #'/home/ec.gpu/Desktop/Soumen/train/ml_zoo_kws/go_12.wav',  on_9.wav   # ',
        help='Audio file to be identified.')
    parser.add_argument(
        '--labels', type=str, default='labels.txt',#'E:/OneDrive - NIT Durgapur/Program/ML_ZOO\ML-zoo-master/models/keyword_spotting/train/cnn_l/validation_utils/labels.txt',
          help='Path to file containing labels.') #/home/ec.gpu/Desktop/Soumen/train/cnn_l/validation_utils/labels.txt'
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
        default='cnn.h5',#'E:/OneDrive - NIT Durgapur/Program/ML_ZOO/ML-zoo-master/models/keyword_spotting/train/cnn_l/keras/cnn.h5',
        help='Path to the .h5 Keras model file to use for testing.')
    FLAGS, unparsed = parser.parse_known_args()
    main()



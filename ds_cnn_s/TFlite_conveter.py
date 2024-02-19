# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 13:28:26 2023

@author: User
"""

import tensorflow as tf
from tensorflow.lite.python import interpreter
from tensorflow import lite
from tensorflow.keras import models
import librosa
import os

audio_dir  = "Dataset"
# Get a list of audio file paths in the directory
list_of_audio_files = [os.path.join(audio_dir, filename) for filename in os.listdir(audio_dir) if filename.endswith(".wav")]

keras_model_filename = 'ds_cnn_s/keras/ds_cnn_s.h5'
tflite_filename = 'ds_cnn_s/keras/ds_cnn_s_quantized.tflite'
tflite_int8_filename = 'ds_cnn_s/keras/ds_cnn_s_int8_quantized.tflite'

# Convert model to TF Lite model
model = models.load_model(keras_model_filename)
converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(tflite_filename, 'wb').write(tflite_model)


# Define the representative dataset generator
def representative_dataset_gen():
    for audio_file_path in list_of_audio_files:
        audio_data, _ = librosa.load(audio_file_path, sr=16000, dtype='int8')
        yield [audio_data]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
########### INT8 ################
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
supported_ops = tf.lite.OpsSet.TFLITE_BUILTINS_INT8
# ########## INT16 ################
# converter.inference_input_type = tf.int16
# converter.inference_output_type = tf.int16
# supported_ops = tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
# ################################
# Int8 post training quantization needs representative dataset.
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [supported_ops]
tflite_INT8_model = converter.convert()
open(tflite_int8_filename, 'wb').write(tflite_INT8_model)

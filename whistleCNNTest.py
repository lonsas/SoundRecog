#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:54:13 2017

@author: alexander
"""

import tensorflow as tf
import ProcessData
import numpy as np
import whistleCNN
import itertools
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.FATAL)
#
#sess=tf.Session()    
##First let's load meta graph and restore weights
#saver = tf.train.import_meta_graph('whistle_convnet_model/model.ckpt-20000.meta')
#saver.restore(sess,tf.train.latest_checkpoint('whistle_convnet_model/'))
#
#
#data, labels, _ = ProcessData.readData()
#data = data.astype(np.float32)
#labels = labels.astype(np.int32)
#eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#  x={"x": data},
#  y=labels,
#  num_epochs=1,
#  shuffle=False)
#eval_results = whistle_classifier.evaluate(input_fn=eval_input_fn)
#print(eval_results)

whistle_classifier = tf.estimator.Estimator(
    model_fn=whistleCNN.cnn_model_fn,
    model_dir="/home/alexander/devel/SoundRecog/whistle_convnet_model_normalized6")

lookup = ['DownWhistle',
          'ForwardCommand',
          'LeftCommand',
          'RightCommand',
          'UpWhistle',
          'background']

def estimate(data):
    data = ProcessData.normalize(data)
    input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": data.astype(np.float32)},
            shuffle=False)
    prediction = whistle_classifier.predict(input_fn=input_fn)
    result = list(itertools.islice(prediction, 1))
    #print(result[0]["probabilities"])
    return lookup[result[0]["classes"]], result[0]["probabilities"][result[0]["classes"]]



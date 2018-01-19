#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 17:08:17 2017

@author: alexander
"""

import pyaudio
import wave
from scipy import signal
from sys import byteorder
from array import array
from struct import pack
import time
import numpy as np
import collections
import SpectrumBuffer as sb
import RingBuffer as rb
import scipy
import whistleCNNTest
import matplotlib.pyplot as plt
import matplotlib
import threading
from pynput.keyboard import Key, Listener, KeyCode
import pickle

record = True;
Labels = ['DownWhistle',
 'ForwardCommand',
 'LeftCommand',
 'RightCommand',
 'UpWhistle',
 'background'];
recordLabel = Labels[5]

def printLabels():
    for i,label in enumerate(Labels):
        print("{0}: {1}".format(i+1,label))
printLabels()

def on_press(key):
    global recordLabel, Labels
    if(recordLabel == Labels[5]):
        if(key == KeyCode.from_char('1')):
            recordLabel = Labels[0]
        elif(key == KeyCode.from_char('2')):
            recordLabel = Labels[1]
        elif(key == KeyCode.from_char('3')):
            recordLabel = Labels[2]        
        elif(key == KeyCode.from_char('4')):
            recordLabel = Labels[3]
        elif(key == KeyCode.from_char('5')):
            recordLabel = Labels[4]
        else:
            recordLabel = Labels[5]
    else:
        recordLabel == Labels[5]
    print('Recording {0}'.format(recordLabel))

def on_release(key):
    global recordLabel, Labels, record
    print('Stopped {0}'.format(recordLabel))
    printLabels()
    recordLabel = Labels[5]
    
    if key == Key.esc:
        # Stop everything
        record = False;
        return False

listener =  Listener(on_press=on_press,
                     on_release=on_release)
listener.start()

import SoundRecogConfig as sc
matplotlib.use('GTKAgg')
plt.axis([0, 50, 0, 50])
    

#plt.ion()

spectrum = sb.SpectrumBuffer(sc.nFrequencies, sc.nTimeSamples)
soundBuffer = rb.RingBuffer(2*sc.dataRate)
background = np.zeros(sc.nFrequencies, dtype='f')
alpha = 0.99;
run = True
prevMean = 0;
detectEnableTicks = 0;
c=0
start_time = time.time()
data = []
dataLabels = []
def callback(in_data, frame_count, time_info, status):
    global spectrum, soundBuffer, recordLabel, data, dataLabels
    snd_data = np.fromstring(in_data, dtype='int16')
    snd_data = signal.decimate(snd_data, sc.downSampleRatio,zero_phase=True);
    soundBuffer.extend(snd_data)
    Y = np.abs(np.fft.rfft(snd_data))[sc.nFrequencyOffsetLow:sc.nFrequencyHighIndex] #Skip lower frequencies
    spectrum.extend(Y)
    curr_spectrum = spectrum.get().T
    data.append(curr_spectrum)
    dataLabels.append(recordLabel)
    return None, pyaudio.paContinue
    

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
               channels=1,
               rate=sc.recordRate,
               input=True,
               frames_per_buffer=sc.chunkSize,
               stream_callback=callback)



stream.start_stream()

while stream.is_active() and record:
    curr_spectrum = spectrum.get().T
    plt.cla()
    plt.pcolormesh(curr_spectrum, vmin=0,cmap='gray');
    plt.pause(0.01)
    #print(whistleCNNTest.estimate(curr_spectrum))
    #print(time.time()-start_time)
    start_time = time.time()

stream.stop_stream()
stream.close()
p.terminate()

dataSet = {"data": data, "labels": dataLabels}
with open('samples/pickles/dataSet_{0}.pkl'.format(time.ctime()), 'wb') as output:
    pickle.dump(dataSet, output, pickle.HIGHEST_PROTOCOL)


    



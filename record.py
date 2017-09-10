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

b = np.load('upwhistle.npy')
b = b/np.std(b)
b = b*2-1

import SoundRecogConfig as sc


p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
               channels=1,
               rate=sc.recordRate,
               input=True,
               frames_per_buffer=sc.chunkSize)

spectrum = sb.SpectrumBuffer(sc.nFrequencies, sc.nTimeSamples)
soundBuffer = rb.RingBuffer(2*sc.dataRate)
background = np.zeros(sc.nFrequencies, dtype='f')
alpha = 0.99;
run = True

start_time = time.time()
while run == True:

    snd_data = np.fromstring(stream.read(sc.chunkSize),dtype='int16')
    
    snd_data = signal.decimate(snd_data, sc.downSampleRatio,zero_phase=True);
    soundBuffer.extend(snd_data)
#    f, t, Zxx = signal.stft(snd_data,
#                            RATE,
#                            nperseg=CHUNK_SIZE/2,
#                            return_onesided=True,
#                            noverlap=0);
    Y = np.abs(np.fft.rfft(snd_data))[sc.nFrequencyOffset:] #SKip lower frequencies
    
    #Filter out background noise i fft domain
    #background = background*alpha+(1-alpha)*Y
    #Y = (Y-background).clip(min=0)

    spectrum.extend(Y)
    
    curr_spectrum = spectrum.get().T
    #
    #curr_spectrum = curr_spectrum/np.max(curr_spectrum)
    #
    #conv = scipy.ndimage.filters.correlate(curr_spectrum,b,mode='constant')
    #conv_zoomed = scipy.ndimage.interpolation.zoom(conv,0.25)
    #conv_zoomed = conv_zoomed - np.mean(conv_zoomed)
    #if(np.max(conv_zoomed) > 3e+5):
    #    print('whistle');
    #else:
    #    print('no whistle');

    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    run = False
stream.stop_stream()
stream.close()
p.terminate()


    



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

import gi  
gi.require_version('Playerctl', '1.0')  
from gi.repository import Playerctl  
player = Playerctl.Player()

import pulsectl
pulse =  pulsectl.Pulse('volume-increaser')


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

def doAction(command):
    global player
    if(command == 'ForwardCommand'):
        player.play_pause()
    elif(command == 'RightCommand'):
        player.next()
    elif(command == 'LeftCommand'):
        player.previous()
    elif(command == 'DownWhistle'):
        for sink in pulse.sink_list():
            pulse.volume_change_all_chans(sink, -0.1)
    elif(command == 'UpWhistle'):
        for sink in pulse.sink_list():
            pulse.volume_change_all_chans(sink, 0.1)    

            

def callback(in_data, frame_count, time_info, status):
    global spectrum, soundBuffer, recordLabel, data, dataLabels
    snd_data = np.fromstring(in_data, dtype='int16')
    snd_data = signal.decimate(snd_data, sc.downSampleRatio,zero_phase=True);
    soundBuffer.extend(snd_data)
    Y = np.abs(np.fft.rfft(snd_data))[sc.nFrequencyOffsetLow:sc.nFrequencyHighIndex] #Skip lower frequencies
    spectrum.extend(Y)
    return None, pyaudio.paContinue
    

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
               channels=1,
               rate=sc.recordRate,
               input=True,
               frames_per_buffer=sc.chunkSize,
               stream_callback=callback)



stream.start_stream()
predictions = [-1]*2
timer = 0
while stream.is_active():
    curr_spectrum = spectrum.get().T
    #plt.cla()
    #plt.pcolormesh(curr_spectrum, vmin=0,cmap='gray');
    #plt.pause(0.1)
    #print(whistleCNNTest.estimate(curr_spectrum))
    #print(time.time()-start_time)
    predictions[1:] = predictions[0:-1]
    prediction, certainty = whistleCNNTest.estimate(curr_spectrum)
    if(certainty > 0.7):
                   #(len(set(predictions)) == 1) and
        predictions[0] = prediction
        if((prediction != -1) and
           (prediction != 'background') and
           (timer > 1)):
            print(certainty)
            print(prediction)
            doAction(prediction)
            timer = 0
    else:
        predictions[0] = -1
    timer+=time.time()-start_time
    start_time = time.time()
    

stream.stop_stream()
stream.close()
p.terminate()

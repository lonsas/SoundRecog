#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 20:22:44 2017

@author: alexander
"""

import scipy
import numpy as np
from scipy import signal
import os
import re
import pickle

import SoundRecogConfig as sc





def createSpectrogram(fileName):

    rate, audioData = scipy.io.wavfile.read(fileName)
    if(rate != sc.recordRate):
        return None
    
    if(audioData.ndim == 2):
        audioData = audioData[:,0]
    audioData = signal.decimate(audioData,sc.downSampleRatio,zero_phase=True);
    if(audioData.size > sc.signalLength):
        return None
    audioData = np.pad(audioData, (0,(sc.signalLength-audioData.size)), 'constant')

    #nframes = file.getnframes();
    #audioData = file.readframes(nframes)
    f, t, Z = signal.stft(audioData, nperseg=sc.dataChunks,return_onesided=True,noverlap=0);
    Z = np.abs(Z[sc.nFrequencyOffset:,1:])
    return Z

def readData():
    labels = []
    data = []
    fileTagR = re.compile(r"[A-Za-z]+")
    for file in os.listdir(sc.samplesDirectory):
        if file.endswith(".wav"):
            
            spectrogram= createSpectrogram(os.path.join(sc.samplesDirectory, file))
            if(spectrogram is not None):
                print(spectrogram.shape)
                label = fileTagR.match(file).group()
                labels.append(label)
                data.append(spectrogram)
    dataSet = {"data": data, "labels": labels}
    return dataSet

if __name__ == "__main__":
    dataSet = readData()
    with open('dataSet.pkl', 'wb') as output:
        pickle.dump(dataSet, output, pickle.HIGHEST_PROTOCOL)
#Zxx = createSpectrum('samples/Processed/UpWhistle0_-02.wav')
#plt.pcolormesh(Zxx, vmin=0,cmap='gray');
#plt.title('STFT Magnitude')
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()

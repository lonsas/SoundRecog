#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 20:22:44 2017

@author: alexander
"""

import scipy
import scipy.io.wavfile
import numpy as np
from scipy import signal
import os
import re
import pickle
import random
import SoundRecogConfig as sc


def normalize(image):
    image -= np.mean(image)
    image /= image.max()    
    return image

def preprocess(data):
    # Normalization
    for image in data:  
        image = normalize(image)
    return data
        
        
      
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
    Z = np.abs(Z[sc.nFrequencyOffsetLow:sc.nFrequencyHighIndex,1:])
    return Z

def setDataRatio(data, numberedLabels, targetLabel, ratio):
    returnData = []
    returnLabels = []
    nFound = 0
    for label in numberedLabels:
        if(label == targetLabel):
            nFound += 1
    nElements = len(numberedLabels)
    currentRatio = nFound/nElements
    if(currentRatio < ratio):
        print("Not enough elements, current ratio is {0}".format(currentRatio))
        nToKeep = nFound
    else:
        #nToKeep/(nBaseElements+nToKeep) = ratio
        #nToKeep= ratio*(nBaseElements+nToKeep)
        #nToKeep = ratio*nBaseElements + ratio*nToKeep
        #nToKeep*(1-ratio) = ratio*nBaseElements
        #nToKeep = ratio*nBaseElement/(1-ratio)
        nToKeep = int(round(ratio*(nElements-nFound)/(1-ratio)))
    print("Keeping {0}/{1} elements of ratiod data ({2}/{3} total elements left)".format(nToKeep, nFound, nElements-nFound+nToKeep, nElements))
    iToKeep = random.sample(range(nFound),nToKeep)
    iToKeepIndex = 0
    foundIndex = 0
    for i,label in enumerate(numberedLabels):
        if(label != targetLabel):
            returnData.append(data[i])
            returnLabels.append(label)
        else:
            if(foundIndex in iToKeep):
                returnData.append(data[i])
                returnLabels.append(label)
                iToKeepIndex+=1              
            foundIndex += 1
    return returnData, returnLabels

def readWave():
    labels = []
    data = []
    fileTagR = re.compile(r"[A-Za-z]+")
    fileList = os.listdir(sc.samplesDirectory);
    fileList.sort()
    for file in fileList:
        if file.endswith(".wav"):
            
            spectrogram = createSpectrogram(os.path.join(sc.samplesDirectory, file))
            if(spectrogram is not None):
                #print(spectrogram.shape)
                label = fileTagR.match(file).group()
                labels.append(label)
                data.append(spectrogram)
    dataArray = np.array([image for image in data])
    return dataArray, labels


def readPickles():
    labels = []
    data = []
    fileList = os.listdir(sc.picklesDirectory);
    for file in fileList:
        if file.endswith(".pkl"):
            with open(os.path.join(sc.picklesDirectory, file), 'rb') as pickle_file:
                dataSet = pickle.load(pickle_file)
                labels.extend(dataSet['labels'])
                data.extend(dataSet['data'])
    return data, labels
    
def readData(backgroundRatio = -1):
    data = []
    labels = []
    dataWav, labelsWaw = readWave()
    data.extend(dataWav)
    labels.extend(labelsWaw)
    
    dataPkl, labelsPkl = readPickles()
    data.extend(dataPkl)
    labels.extend(labelsPkl)
    
    _, numberedLabels = np.unique(labels, return_inverse=True)
    lookup = [None]*(numberedLabels.max()+1)
    for i,d in enumerate(numberedLabels):
        if(d not in lookup):
            lookup[d] = labels[i]
    if(backgroundRatio >= 0):
        data, numberedLabels = setDataRatio(data, numberedLabels, lookup.index("background"), backgroundRatio)
    dataArray = np.array([image for image in data])
    numberedLabelsArray = np.array(numberedLabels)
    return dataArray, numberedLabelsArray, lookup




if __name__ == "__main__":
    data, labels, _ = readData(0.2)
    dataSet = {"data": data, "labels": labels}
    with open('dataSet_all.pkl', 'wb') as output:
        pickle.dump(dataSet, output, pickle.HIGHEST_PROTOCOL)
        
#Zxx = createSpectrum('samples/Processed/UpWhistle0_-02.wav')
#plt.pcolormesh(Zxx, vmin=0,cmap='gray');
#plt.title('STFT Magnitude')
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()

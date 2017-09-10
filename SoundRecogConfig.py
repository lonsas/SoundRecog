#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 20:30:18 2017

@author: alexander
"""

recordRate = 44100
downSampleRatio = 7
dataChunks = 128;
spectrumTime = 2
nFrequencyOffset = 15 #Lower frequencies to skip



dataRate = int(recordRate/downSampleRatio)
nFrequencies = int(dataChunks/2-nFrequencyOffset+1)
nTimeSamples = int(50) #spectrumTime*recordRate/downSampleRatio/dataChunks
chunkSize= dataChunks*downSampleRatio;
signalLength = nTimeSamples*dataChunks


samplesDirectory = 'samples/Processed/'
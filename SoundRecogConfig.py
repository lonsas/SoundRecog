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
nFrequencyOffsetLow = 5 #Lower frequencies to skip
nFrequencyOffsetHigh = 10 #High frequencies to skip



dataRate = int(recordRate/downSampleRatio)
nFrequencies = int(dataChunks/2-nFrequencyOffsetLow-nFrequencyOffsetHigh+1)
nFrequencyHighIndex = nFrequencies+nFrequencyOffsetLow
nTimeSamples = int(50) #spectrumTime*recordRate/downSampleRatio/dataChunks
chunkSize= dataChunks*downSampleRatio;
signalLength = nTimeSamples*dataChunks


samplesDirectory = 'samples/Processed/'
picklesDirectory = 'samples/pickles/'
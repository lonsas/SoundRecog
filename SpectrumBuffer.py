#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 18:27:39 2017

@author: alexander
"""
import numpy as np

class SpectrumBuffer():
    "A 2D ring buffer using numpy arrays"
    def __init__(self, x, y):
        self.data = np.zeros([x, y], dtype='f')
        self.index = 0
        self.buffSize = x

    def extend(self, x):
        "adds array x to ring buffer"
        x_index = self.index
        self.data[x_index] = x
        self.index = (x_index + 1) % self.buffSize

    def get(self):
        "Returns the first-in-first-out data in the ring buffer"
        idx = (self.index + np.arange(self.buffSize)) %self.buffSize
        return self.data[idx]
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:26:19 2020

@author: dacole2
"""
from abc import ABCMeta, abstractmethod

class InputDistribution(metaclass = ABCMeta):
    def __init__(self):
        pass
    
    @abstractmethod
    def draw_samples(self, num_samples):
        pass
    @abstractmethod    
    def evaluate_pdf(self, samples):
        pass
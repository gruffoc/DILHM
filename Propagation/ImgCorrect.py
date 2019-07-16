#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 16:21:39 2019

@author: claudriel

Script for correct the raw-images from the background. 
The programs take all the N images in the directory path and it calculates 
the median through the function "median" in "__function"
At each set of b images, it subtracts the respective median to the images and 
it save them in another folder.
"""
import sys
import os
import numpy as np
import holopy as hp
import matplotlib.pyplot as plt

from __function import *

###### Path name
name = "Poly/1um/7"
directory_sample = "../Campioni/Flusso/"+name+"/dati/"
directory_save_bg = "../Campioni/Flusso/"+name+"/mediana/img_"   
directory_save_correct = "../Campioni/Flusso/"+name+"/img_correct/img_"

##### N: numbers of image, a-b interval between making the median
a = 0
b = 10
N = 50
    
mediana(directory_sample, directory_save_bg, directory_save_correct, a, b, N)
print('Le mediane sono state calcolate e le immagini corrette')
    

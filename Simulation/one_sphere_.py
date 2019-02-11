#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:44:25 2019

@author: claudiaravasio
"""

import holopy as hp
from holopy.scattering import calc_holo, Sphere

sphere = Sphere(n = 1.59, r = 0.3, center = (25.6, 25.6, 7))
medium_index = 1.33
illum_wavelen = 0.660
illum_polarization = (1,0)
detector = hp.detector_grid(shape = 512, spacing = 0.1)

holo = calc_holo(detector, sphere, medium_index, illum_wavelen, illum_polarization)
hp.show(holo)
#hp.save_image('',holo)
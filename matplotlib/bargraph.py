#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:47:41 2022

@author: akhilgogineni
"""

import matplotlib.pyplot as plt
import numpy as np

population = [1415, 1354, 326, 266, 210, 200, 195, 166, 144, 130]
countries = ['China', 'India', 'US', 'Indonesia', 'Brazil', 'Pakistan', 'Nigeria', 'Bangladesh', 'Russia', 'Mexico']

xpos = np.arange(len(countries))
plt.xticks(xpos, countries, rotation = 'horizontal')
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 4
plt.rcParams["figure.figsize"] = fig_size

plt.ylabel("population in millions")
plt.title("Population per country")
plt.bar(xpos, population, data = "countries", color = "#3c5cb5", width=0.8)
plt.legend(loc="upper right")


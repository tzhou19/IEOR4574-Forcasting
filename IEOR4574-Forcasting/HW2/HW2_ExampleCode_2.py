# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 18:21:47 2022

@author: waseem
"""
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acovf
from matplotlib.ticker import FormatStrFormatter
import datetime as dt
from scipy import signal
from scipy.fft import fft, ifft
from matplotlib.ticker import AutoMinorLocator
import math
from statsmodels.formula.api import ols
from statsmodels.graphics.api import interaction_plot, abline_plot
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm


# dv = pd.read_excel('Viscosity.xlsx')
# dw = pd.read_excel('WholeFood.xlsx')
# dp = pd.read_excel('Pharmaceutical.xlsx')


# Autocorrelation 
dicfn = pd.read_csv('IPN31152N_Fred_Industrial Production Manufacturing Non-Durable Goods Ice Cream and Frozen Dessert.csv')
dicfn.DATE = pd.to_datetime(dicfn.DATE)
plt.plot(dicfn.DATE[dicfn.DATE.dt.year > 2015],dicfn.IPN31152N[dicfn.DATE.dt.year > 2015])
plt.ylabel('Production Index')
plt.xlabel('Time t')
plt.legend(['Icream Daily Production'])
plt.grid(True)
t_date = dicfn.DATE[dicfn.DATE.dt.year > 2015]
x = dicfn.IPN31152N[dicfn.DATE.dt.year > 2015].values

plt.figure(figsize=(10, 7))
fig, ax = plt.subplots()
ax.acorr(x,maxlags = 30)
minor_locator = AutoMinorLocator(10)
ax.xaxis.set_minor_locator(minor_locator)
plt.grid(which='minor')
plt.legend(['$R_X$(\u03C4)'],loc='upper left')
plt.xlabel('\u03C4')
# plt.ylabel('Power')
ax.grid(True, which='both')
plt.tight_layout()
fig.savefig('Stochastic_Autocorrelation.png')
fig.show()

#PSD
freqs, psd = signal.welch(x)
plt.figure(figsize=(10, 7))
fig, ax = plt.subplots()
ax.plot(freqs, psd)
minor_locator = AutoMinorLocator(5)
ax.xaxis.set_minor_locator(minor_locator)
plt.grid(which='minor')
plt.legend(['PSD'])
plt.xlabel('Frequency')
plt.ylabel('Power')
ax.grid(True, which='both')
plt.tight_layout()
fig.savefig('Stochastic_PSD.png')
fig.show()

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:40:28 2022

@author: waseem
"""

import numpy as np
import matplotlib.pyplot as plt  # To visualize


### Stochastic process
t = np.arange(0, 100)
s1 = np.random.normal(4,2,100)
s2 = np.random.normal(4,2,100)
s3 = np.random.normal(4,2,100)
s4 = np.random.normal(4,2,100)

fig = plt.figure(figsize=(20, 14))

ax1 = plt.subplot(411)
plt.plot(t, s1)
plt.xlim(0, 100)
plt.ylim(0, 10)
plt.ylabel('$X_1$(t)',fontsize=20)
plt.tick_params('x', labelbottom=False)
plt.yticks(fontsize=12)
ax1.xaxis.grid(linewidth=2)
ax1.yaxis.grid(linewidth=1)

# share x only
ax2 = plt.subplot(412, sharex=ax1)
plt.plot(t, s2)
plt.xlim(0, 100)
plt.ylim(0, 10)
plt.ylabel('$X_2$(t)',fontsize=20)
# make these tick labels invisible
plt.tick_params('x', labelbottom=False)
plt.yticks(fontsize=12)
ax2.xaxis.grid(linewidth=2)
ax2.yaxis.grid(linewidth=1)

# share x only
ax3 = plt.subplot(413, sharex=ax1)
plt.plot(t, s3)
plt.xlim(0, 100)
plt.ylim(0, 10)
plt.ylabel('$X_3$(t)',fontsize=20)
# make these tick labels invisible
plt.tick_params('x', labelbottom=False)
plt.yticks(fontsize=12)
ax3.xaxis.grid(linewidth=2)
ax3.yaxis.grid(linewidth=1)


# share x and y
ax4 = plt.subplot(414, sharex=ax1, sharey=ax1)
plt.plot(t, s4)
plt.xlim(0, 100)
plt.ylim(0, 10)
plt.ylabel('$X_4$(t)',fontsize=20)
ax4.xaxis.grid(linewidth=2)
ax4.yaxis.grid(linewidth=1)
plt.xticks(fontsize=20)
# plt.savefig('Stochastic_Processes.png')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import json
import os

shade = ['red', 'yellow', 'blue', 'grey', 'orange',
         'purple', 'maroon', 'green', 'darkgoldenrod', 'steelblue']
issues = ['Pedestal Shift', 'Data Spike',
          'Insufficient Data', 'Stuck Bit', 'None']

os.chdir(r'D:\14GeV')

# This is to load up the problem tiles.
try:
    pedshift = np.loadtxt('pedshift.txt')
except OSError:
    pedshift = np.zeros((2, 5))
try:
    spikes = np.loadtxt('spikes.txt')
except OSError:
    spikes = np.zeros((2, 5))
try:
    indata = np.loadtxt('indata.txt')
except OSError:
    indata = np.zeros((2, 5))
try:
    stbit = np.loadtxt('stbit.txt')
except OSError:
    stbit = np.zeros((2, 5))
pedshift[:, 4] = 0
spikes[:, 4] = 1
indata[:, 4] = 2
stbit[:, 4] = 3
pedshift = pedshift.astype(int)
spikes = spikes.astype(int)
indata = indata.astype(int)
stbit = stbit.astype(int)

shader = [[0, 4, 5, 6], [4, 1, 7, 8], [5, 7, 2, 9], [6, 8, 9, 3]]

x = []
y = []

comp0 = []
comp1 = []
comp2 = []
comp3 = []

for i in pedshift[:, 1:-1]:
    comp0.append(str(i))
for i in spikes[:, 1:-1]:
    comp1.append(str(i))
for i in indata[:, 1:-1]:
    comp2.append(str(i))
for i in stbit[:, 1:-1]:
    comp3.append(str(i))

if pedshift[0][0] != 0:
    for i in pedshift[:, 1:-1]:
        a = 0
        if str(i) in comp1:
            a = shader[0][1]
        if str(i) in comp2:
            a = shader[0][2]
        if str(i) in comp3:
            a = shader[0][3]
        if i[0] == 0:
            x.append(np.append(i[1:], a))
        if i[0] == 1:
            y.append(np.append(i[1:], a))

if spikes[0][0] != 0:
    for i in spikes[:, 1:-1]:
        a = 1
        if str(i) in comp0:
            a = shader[1][0]
        if str(i) in comp2:
            a = shader[1][2]
        if str(i) in comp3:
            a = shader[1][3]
        if i[0] == 0:
            x.append(np.append(i[1:], a))
        if i[0] == 1:
            y.append(np.append(i[1:], a))

if indata[0][0] != 0:
    for i in indata[:, 1:-1]:
        a = 2
        if str(i) in comp0:
            a = shader[2][0]
        if str(i) in comp1:
            a = shader[2][1]
        if str(i) in comp3:
            a = shader[2][3]
        if indata[i][0] == 0:
            x.append(np.append(indata[i][1:], a))
        if indata[i][0] == 1:
            y.append(np.append(indata[i][1:], a))

if stbit[0][0] != 0:
    for i in stbit[:, 1:-1]:
        a = 3
        if str(i) in comp0:
            a = shader[3][0]
        if str(i) in comp2:
            a = shader[3][2]
        if str(i) in comp1:
            a = shader[3][1]
        if stbit[i][0] == 0:
            x.append(np.append(stbit[i][1:], a))
        if stbit[i][0] == 1:
            y.append(np.append(stbit[i][1:], a))

x, y = np.asarray(x), np.asarray(y)

fig = plt.figure(figsize=(15, 10), facecolor='white')
ax = fig.add_subplot(111)
rrange = np.arange(18)
labels = [' ', '3', ' ', '2', ' ', '1', ' ', '12', ' ', '11', ' ',
          '10', ' ', '9', ' ', '8', ' ', '7', ' ', '6', ' ', '5', ' ', '4']

theta = []
for i in range(24):
    theta.append(np.arange((np.pi/12)*i, (np.pi/12)*(i+1), np.pi/360))
theta1 = []
for i in range(12):
    theta1.append(np.arange((np.pi/6)*i, (np.pi/6)*(i+1), np.pi/360))

# East side of the EPD.
plt.polar()
plt.thetagrids([theta*15 for theta in range(360//15)], labels, fontsize=20)
for i in range(int(len(theta))):
    plt.fill_between(theta[i], 1, 17, color='wheat')
    for j in theta[i]:
        plt.polar([j, j+np.pi/720], [17, 17], color='black', lw=3)
        plt.polar([j, j+np.pi/720], [1, 1], color='black', lw=3)
for i in x:
    pos = 2*(3-i[0]) % 24
    pos1 = int((3-i[0]) % 12)
    half = (i[1]+1) % 2
    index = int(pos+half)
    ran = int(i[1]/2)
    if i[1] == 1:
        plt.fill_between(theta1[pos1], ran+1, ran+2, color=shade[int(i[2])])
    else:
        plt.fill_between(theta[index], ran+1, ran+2, color=shade[int(i[2])])
ranger = np.arange(0, 2*np.pi, np.pi/6)
for i in ranger:
    plt.polar([i, i], [1, 17], color='black', lw=3)
plt.rgrids(rrange[:-1])
plt.title("East Side EPD", fontsize=30)
leg = plt.legend(issues, fontsize=20, loc=1)
for i in range(int(len(issues))):
    leg.legendHandles[i].set_color(shade[i])
    leg.legendHandles[i].set_linewidth(8.0)
leg.legendHandles[-1].set_color('wheat')
leg.legendHandles[-1].set_linewidth(8.0)
# All this noise is simply to offset the legend.
plt.draw()
bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
xOffset = 0.1
bb.x0 += xOffset
bb.x1 += xOffset
leg.set_bbox_to_anchor(bb, transform=ax.transAxes)

plt.show()
plt.close()

# West side of the EPD.
fig = plt.figure(figsize=(15, 10), facecolor='white')
ax = fig.add_subplot(111)

plt.polar()
plt.thetagrids([theta*15 for theta in range(360//15)], labels, fontsize=20)
for i in range(int(len(theta))):
    plt.fill_between(theta[i], 1, 17, color='wheat')
    for j in theta[i]:
        plt.polar([j, j+np.pi/720], [17, 17], color='black', lw=3)
        plt.polar([j, j+np.pi/720], [1, 1], color='black', lw=3)
for i in y:
    pos = 2*(3-i[0]) % 24
    pos1 = int((3-i[0]) % 12)
    half = (i[1]+1) % 2
    index = int(pos+half)
    ran = int(i[1]/2)
    if i[1] == 1:
        plt.fill_between(theta1[pos1], ran+1, ran+2, color=shade[int(i[2])])
    else:
        plt.fill_between(theta[index], ran+1, ran+2, color=shade[int(i[2])])
ranger = np.arange(0, 2*np.pi, np.pi/6)
for i in ranger:
    plt.polar([i, i], [1, 17], color='black', lw=3)
plt.rgrids(rrange[:-1])
plt.title("West Side EPD", fontsize=30)
leg = plt.legend(issues, fontsize=20, loc=1)
for i in range(int(len(issues))):
    leg.legendHandles[i].set_color(shade[i])
    leg.legendHandles[i].set_linewidth(8.0)
leg.legendHandles[-1].set_color('wheat')
leg.legendHandles[-1].set_linewidth(8.0)
# All this noise is simply to offset the legend.
plt.draw()
bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
xOffset = 0.1
bb.x0 += xOffset
bb.x1 += xOffset
leg.set_bbox_to_anchor(bb, transform=ax.transAxes)

plt.show()
plt.close()

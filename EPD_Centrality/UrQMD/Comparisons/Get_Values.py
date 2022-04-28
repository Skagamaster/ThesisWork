import numpy as np
import uproot as up
import matplotlib.pyplot as plt
import os

# Let's do something new.
energy = str(200)
col = energy
if energy == '19':
    col = '19.6'
if energy == '7':
    col = '7.7'
os.chdir(r'D:\UrQMD_cent_sim')
data = up.open('hist_ppkset_rap_%s.root' % energy)
dkeys = np.asarray(data.keys()[1:])
dl = int(len(dkeys))
cl = int(len(data[data.keys()[1]][1:-1]))

bVarsRosi = []

for i in range(dl):
    bVarsRosi.append(data[dkeys[i]][1:-1])
print(bVarsRosi)
bVarsRosi = np.asarray(bVarsRosi)
np.savetxt('bVarsRosi%s.txt' % energy, bVarsRosi)

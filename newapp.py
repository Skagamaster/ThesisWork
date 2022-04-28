import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os



os.chdir(r'D:\14GeV\Stds_Picos')
aves = []
e_set = [3, 4, 5]
t_set = [6, 7, 8, 9, 10]
for i in range(32):
    e_set.append(i+11)
count = 0
for i in os.listdir():
    if i.endswith('txt'):
        aves.append([])
        arr = np.loadtxt(i)
        aves[count].append(arr[0])
        events = arr[1]
        tracks = arr[2]
        for j in e_set:
            aves[count].append(arr[j]/events)
        for j in t_set:
            aves[count].append(arr[j]/tracks)
        count += 1
aves = np.asarray(aves)
np.savetxt(r'D:\14GeV\fo_pico_aves.txt', aves, delimiter=',', newline='},{', fmt='%f')
print(np.shape(aves))

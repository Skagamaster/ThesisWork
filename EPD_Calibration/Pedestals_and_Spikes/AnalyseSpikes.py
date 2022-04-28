import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def Directory1(day):
    day = str(day)
    direct = r'D:\14GeV\Day%s\Runs\RunPeds' % day
    os.chdir(direct)


Directory1(110)
values = []
names = []
for i in os.listdir():
    if Path(i).suffix != '.npy':
        continue
    values.append(np.load(i, allow_pickle=True))
    names.append(int(str(i)[:-9]))

values = np.asarray(values)
plt.plot(values[0].flatten())
plt.show()

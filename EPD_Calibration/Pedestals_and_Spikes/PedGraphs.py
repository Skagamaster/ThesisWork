import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


def Directory(day):
    day = str(day)
    #direct = r'D:\14GeV\Day%s\Runs\RunPeds' % day
    direct = r'C:\PhysicsProcessing\7.7_Ped_check\Day%s\RunPeds' % day
    os.chdir(direct)


def EPDTile(tile=int(123)):
    if tile < 0:
        ew = 0
    else:
        ew = 1
    pp = int(abs(tile/100))
    tt = abs(tile) % 100
    return [ew, pp, tt]


days = [157, 158]

pedestals = []
names = []
for i in days:
    try:
        Directory(i)
    except FileNotFoundError:
        continue
    for j in os.listdir():
        if Path(j).suffix != '.npy':
            continue
        pedestals.append(np.load(j, allow_pickle=True).flatten())
        names.append(str(j)[:-9])

pedestals = np.asarray(pedestals)
names = np.asarray(names)

# os.chdir(r'D:\14GeV')
os.chdir(r'C:\PhysicsProcessing\7.7_Ped_check')
np.save('pedestals.npy', pedestals)
np.save("pednames.npy", names)

x = []
y = []
z = []
countX = 0
countY = 0
for i in pedestals:
    for j in i:
        x.append(countX)
        y.append(countY)
        z.append(j)
        countY += 1
    countX += 1
    countY = 0

fig = plt.figure(figsize=(10, 8))
ax = fig.gca(projection='3d')
surf = ax.plot_trisurf(x, y, z, cmap="jet", linewidth=0.2)  # , vmax=500)
#fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_zlim(0, 200)
ax.set_ylabel("Tile Index", fontsize=20)
ax.set_xlabel("Run Index", fontsize=20)
ax.set_zlabel("Pedestal Shift", fontsize=20)
plt.show()
plt.show()

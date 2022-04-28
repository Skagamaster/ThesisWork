import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.text import TextPath
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d import Axes3D

# set the first and last runs:
Min = 20109001
Max = 20150007

# Import the abberrant tile data.
os.chdir(r'D:\14GeV')
data = []
for i in range(110, 150):
    for j in os.listdir():
        if j == f'problems{i}.txt':
            data.append(np.loadtxt(j))

data.append(np.loadtxt('problemsnew.txt'))

# Set the colors.
shade = ['red', 'yellow', 'blue', 'grey', 'orange',
         'purple', 'maroon', 'green', 'darkgoldenrod', 'steelblue']
issues = ['Pedestal Shift', 'Data Spike',
          'Insufficient Data', 'Stuck Bit']

x = []
y = []
z = []
color = []

for i in data:
    for j in i:
        if j[1] == 1:
            continue
        r = int(j[3]/2)+1
        theta = np.pi/2 + (j[2]-1)*np.pi/6 + np.pi/24 + ((j[3]+1) % 2)*np.pi/12
        if j[3] == 1:
            theta += np.pi/24
        x.append(r*np.cos(theta))
        y.append(r*np.sin(theta))
        z.append(j[0])
        color.append(shade[int(j[4])])
x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)

# This is for plotting the text on the graph.
# Credit: https://matplotlib.org/3.2.1/gallery/mplot3d/pathpatch3d.html


def text3d(ax, xyz, s, zdir="z", size=None, angle=0, usetex=False, **kwargs):
    '''
    Plots the string 's' on the axes 'ax', with position 'xyz', size 'size',
    and rotation angle 'angle'.  'zdir' gives the axis which is to be treated
    as the third dimension.  usetex is a boolean indicating whether the string
    should be interpreted as latex or not.  Any additional keyword arguments
    are passed on to transform_path.

    Note: zdir affects the interpretation of xyz.
    '''
    x, y, z = xyz
    if zdir == "y":
        xy1, z1 = (x, z), y
    elif zdir == "x":
        xy1, z1 = (y, z), x
    else:
        xy1, z1 = (x, y), z

    text_path = TextPath((0, 0), s, size=size, usetex=usetex)
    trans = Affine2D().rotate(angle).translate(xy1[0], xy1[1])

    p1 = PathPatch(trans.transform_path(text_path), **kwargs)
    ax.add_patch(p1)
    art3d.pathpatch_2d_to_3d(p1, z=z1, zdir=zdir)

# This is to graph the East side of the EPD (E and W PP run in different directions).


def plotIssues(data, Min, Max, shade, issues, azim, elev, off, ew):
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    x = []
    y = []
    z = []
    color = []

    for i in data:
        for j in i:
            if ew == 'w':
                if j[1] == 1:
                    continue
                r = int(j[3]/2)+1
                theta = np.pi/2 + (j[2]-1)*np.pi/6 + np.pi / \
                    24 + ((j[3]+1) % 2)*np.pi/12
                if j[3] == 1:
                    theta += np.pi/24
                x.append(r*np.cos(theta))
                y.append(r*np.sin(theta))
                z.append(j[0])
                color.append(shade[int(j[4])])
            if ew == 'e':
                if j[1] == 0:
                    continue
                r = int(j[3]/2)+1
                theta = np.pi/2 - (j[2]-1)*np.pi/6 - np.pi / \
                    24 - ((j[3]+1) % 2)*np.pi/12
                if j[3] == 1:
                    theta -= np.pi/24
                x.append(r*np.cos(theta))
                y.append(r*np.sin(theta))
                z.append(j[0])
                color.append(shade[int(j[4])])
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    ax.scatter(x, y, z, s=20, c=color, marker='o', alpha=0.7)
    if ew == 'w':
        for i in range(17):
            for j in range(12):
                text3d(ax, (20*np.cos(7*np.pi/12+j*2*np.pi/12), 20*np.sin(7*np.pi/12+j*2*np.pi/12), Max),
                       f'{j+1}',
                       zdir="z", size=2, usetex=False,
                       ec="none", fc="k")
    if ew == 'e':
        for i in range(17):
            for j in range(12):
                text3d(ax, (20*np.cos(5*np.pi/12-j*2*np.pi/12), 20*np.sin(5*np.pi/12-j*2*np.pi/12), Max),
                       f'{j+1}',
                       zdir="z", size=2, usetex=False,
                       ec="none", fc="k")
    for i in range(17):
        p = Circle((0, 0), i, fill=False, lw=0.3)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=Max, zdir="z")
    for i in range(6):
        t = (np.pi/6)*i
        x1 = [0, 16*np.cos(t)]
        y1 = [0, 16*np.sin(t)]
        y2 = [0, -16*np.sin(t)]
        x2 = [0, -16*np.cos(t)]
        plt.plot(x1, y1, Max, c='k')
        plt.plot(x2, y2, Max, c='k')
        plt.plot(x1, y1, Min, c='k')
        plt.plot(x2, y2, Min, c='k')
    leg = ax.legend(issues)
    for i in range(int(len(issues))):
        leg.legendHandles[i].set_color(shade[i])
        leg.legendHandles[i].set_linewidth(8.0)
    if off == True:
        ax.set_axis_off()
    ax.view_init(azim=azim, elev=elev)
    if ew == 'e':
        plt.title('EPD East', fontsize=30)
    if ew == 'w':
        plt.title('EPD West', fontsize=30)
    plt.show()
    plt.close()


plotIssues(data, Min, Max, shade, issues, 270, 90, True, 'e')
plotIssues(data, Min, Max, shade, issues, 220, 20, False, 'e')
plotIssues(data, Min, Max, shade, issues, 270, 90, True, 'w')
plotIssues(data, Min, Max, shade, issues, 220, 20, False, 'w')

# Now to plot West.

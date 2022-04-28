import uproot as up
import os
import numpy as np
import simplejson
from pathlib import Path
from scipy import stats

# Put your directory names here; the first is for running
# over Runs in a day, the second for storing the pedestal
# values.


def Directory(day):
    day = str(day)
    #direct = r'D:\14GeV\Day%s\Runs' % day
    direct = r'C:\PhysicsProcessing\7.7_Ped_check\Day%s' % day
    os.chdir(direct)


def Directory1(day):
    day = str(day)
    #direct = r'D:\14GeV\Day%s\Runs\RunPeds' % day
    direct = r'C:\PhysicsProcessing\7.7_Ped_check\Day%s\RunPeds' % day
    os.chdir(direct)

# This finds the maximum value for each run and stores it
# in a numpy array for analysis later.


def Find_Max(histo):
    maxVals = []
    for i in histo.keys():
        arr = np.asarray(histo[i].numpy()[0])
        tileMaxes = np.asarray(np.where(arr == np.max(arr))).flatten()[0]
        if np.sum(arr) > 200:
            maxVals.append(tileMaxes)
        else:
            maxVals.append(0)
    maxVals = (np.asarray(maxVals)).reshape(2, 12, 31)
    return maxVals


# Enter the days in your energy you want to run over.
for i in range(157, 159):
    Directory(i)
    print('Day', i)
    tracker = 1

    # This portion imports the ROOT histograms as python arrays.
    print('Importing file:')
    files = []
    strFiles = []
    for j in os.listdir():
        if tracker % 5 == 0:
            print(tracker, 'of', len(os.listdir()))
        if Path(j).suffix != '.root':
            tracker += 1
            continue
        files.append(up.open(j))
        strFiles.append(str(j))
        tracker += 1

    # This portion returns the maxes from the ROOT histos, per tile.
    # Structure is [2][12][31] arrays; note the second two are
    # pp-1 and tt-1, so mind that when analysing.
    maxes = []
    tracker = 1
    Directory1(i)
    print('Analysing file:')
    for j in range(int(len(files))):
        if tracker % 5 == 0:
            print(tracker, 'of', len(files))
        histo = files[j]
        tracker += 1
        values = Find_Max(histo)
        np.save(strFiles[j]+'.npy', values)
        f = open(strFiles[j]+'.txt', 'w')
        simplejson.dump(values.tolist(), f)

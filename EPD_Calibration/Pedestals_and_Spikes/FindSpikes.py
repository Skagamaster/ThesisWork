import uproot as up
import os
import numpy as np
import simplejson
from pathlib import Path
from scipy import stats


def Directory(date):
    day = str(date)
    direct = r'D:\2020Picos\9p2GeV\%s' % day
    os.chdir(direct)


def Find_Max(histo):
    values = []
    where = []
    for i in histo.keys():
        arr = np.asarray(histo[i].numpy()[0])
        values.append(np.max(arr[20:]))
        here = np.asarray(np.where(arr == np.max(arr)))
        here = np.concatenate(here)
        where.append(here[0])
    values = np.asarray(values)
    where = np.asarray(where)
    return values, where


for i in range(204, 206):
    Directory(i)
    print('Day', i)
    # Importing the files into an array.
    tracker = 1
    print('Importing file:')
    files = []
    for i in os.listdir():
        if tracker % 5 == 0:
            print(tracker, 'of', len(os.listdir()))
        if Path(i).suffix != '.root':
            tracker += 1
            continue
        files.append(up.open(i))
        tracker += 1

    # Processing the files and extracting the tile maxes.
    vals = []
    whr = []
    tracker = 1
    print('Analysing file:')
    for i in range(int(len(files))):
        if tracker % 5 == 0:
            print(tracker, 'of', len(files))
        histo = files[i]
        tracker += 1
        values, where = Find_Max(histo)
        vals.append(values)
        whr.append(where)
    vals = np.asarray(vals)
    whr = np.asarray(whr)
    np.save('npvals.npy', vals)
    np.save('npwhere.npy', whr)

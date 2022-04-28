import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import simplejson
import pandas as pd


def Directory(date):
    day = str(date)
    direct = r'D:\2020Picos\9p2GeV\Day%s' % day
    os.chdir(direct)


for i in range(204, 206):
    Directory(i)
    print("Day", i)
    # This is to get a baseline estimate for the 1st MIP MPV.
    # File should be a single day's worth of calibrated values.
    # Format: day   ew  pp  tt  nmip    err
    nmip = np.loadtxt(r'D:\14GeV\nmipSpike.txt')
    nmip = nmip[:, 1:-1]  # <- strips day and err

    values = np.load('npvals.npy', allow_pickle=True)
    where = np.load('npwhere.npy', allow_pickle=True)
    runs = []
    for j in os.listdir():
        if Path(j).suffix != '.root':
            continue
        runs.append(str(j))
    nodata = []
    badpos = []
    pdpos = []
    pdrun = []
    for k in range(int(len(values))):
        # pdrun.append(int(runs[k][:-5]))
        print(runs[k][:-5])
        pdrun.append(int(runs[k][:-5]))
        pdpos.append([])
        for j in range(int(len(values[k]))):
            ew = int(j/372)
            pp = int(j/31)+1-ew*12
            tt = (j % 31)+1
            if values[k][j] == 0.0:
                if runs[k] not in nodata:
                    nodata.append(runs[k])
            if where[k][j] > 25 and values[k][j] > 100:
                # if runs[k] not in badpos:
                    # badpos.append(runs[k])
                # badpos.append([int(runs[k][:8]), ew, pp, tt, 0])
                badpos.append([int(runs[k][:-5]), ew, pp, tt, 0])
                pdpos[k].append([ew, pp, tt, where[k][j]])

    df = pd.DataFrame(pdpos, index=pdrun)
    badlist = []

    for k in range(int(len(df))):
        name = df.index[k]
        for j in range(int((len(df.iloc[k])))):
            if df.iloc[k][j] != None:
                entry = 372*df.iloc[k][j][0]+31 * \
                    (df.iloc[k][j][1]-1)+df.iloc[k][j][2]-1
                comp = df.iloc[k][j][3]/nmip[entry][3]
                if comp > 0.3:
                    # if name not in badlist:
                    #    badlist.append(name)
                    badlist.append(df.iloc[k][j][:-1])

    print('Runs without any data:', nodata)
    print('Spikes to look into:', badpos)

    # Save our data for analysis later, if needed.
    # os.chdir(r'F:\19.6gev_new')
    badpos = np.asarray(badpos)
    np.save(f'problems{i}.npy', badpos)
    np.savetxt(f'problems{i}.txt', badpos)
    f = open(f'nodata{i}.txt', 'w')
    h = open(f'badlist{i}.txt', 'w')
    df.to_pickle(f'df{i}.pkl')
    simplejson.dump(nodata, f)
    simplejson.dump(badlist, h)
    f.close()
    h.close()

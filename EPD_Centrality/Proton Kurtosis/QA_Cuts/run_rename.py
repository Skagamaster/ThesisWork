import os
import numpy as np
import uproot as up
import awkward as ak

dataDirect = r'E:\2019Picos\14p5GeV\Runs'
os.chdir(dataDirect)
files = os.listdir()
for file in files:
    try:
        run_id = 0
        with up.open(file) as data:
            data = data['PicoDst']['Event']['Event.mRunId'].array(library='np')
            run_id = int(data[0])
        print(run_id)
        newfile = str(run_id) + r'.picoDst.root'
        os.rename(file, newfile)
    except Exception as e:
        print(file, "didn't have a PicoDst branch.")

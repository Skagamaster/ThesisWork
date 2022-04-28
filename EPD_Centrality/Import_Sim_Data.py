import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

os.chdir(r'D:\UrQMD_cent_sim\27')
data = np.loadtxt('arrayUrQMD.txt')

# Save it all as a Pandas dataframe.
columns = ['b', 'refMult', 'ring01', 'ring02', 'ring03', 'ring04', 'ring05', 'ring06', 'ring07', 'ring08',
           'ring09', 'ring10', 'ring11', 'ring12', 'ring13', 'ring14', 'ring15', 'ring16']
dArray = []
for i in range(18):
    dArray.append(data[:, i])
dArray = np.asarray(dArray)
rows = np.linspace(0, len(data)-1, len(data), dtype=int)
df = pd.DataFrame(data=dArray.T, index=rows, columns=columns)
df.to_pickle('pandassim.pkl')

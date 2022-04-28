import os
import numpy as np
import matplotlib.pyplot as plt

# Load the newest numbers.
""""""
os.chdir(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_4_14_2021\protons\total")
pro_high = np.load("high_proton_ave.npy", allow_pickle=True)
apro_high = np.load("high_antiproton_ave.npy", allow_pickle=True)
pro_low = np.load("low_proton_ave.npy", allow_pickle=True)
apro_low = np.load("low_antiproton_ave.npy", allow_pickle=True)
runs = np.load("runs.npy", allow_pickle=True).astype('int')

# Load the existing numbers.
os.chdir(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_4_14_2021\protons\total")
"""
pro_high1 = np.load("high_proton_ave.npy", allow_pickle=True)
apro_high1 = np.load("high_antiproton_ave.npy", allow_pickle=True)
pro_low1 = np.load("low_proton_ave.npy", allow_pickle=True)
apro_low1 = np.load("low_antiproton_ave.npy", allow_pickle=True)
runs1 = np.load("runs.npy", allow_pickle=True).astype('int')
# Consolidate the two lists.
pro_high = np.hstack((pro_high1, pro_high))
pro_low = np.hstack((pro_low1, pro_low))
apro_high = np.hstack((apro_high1, apro_high))
apro_low = np.hstack((apro_low1, apro_low))
runs = np.hstack((runs1, runs)).astype('int')
# Save the result.
np.save("high_proton_ave.npy", pro_high)
np.save("high_antiproton_ave.npy", apro_high)
np.save("low_proton_ave.npy", pro_low)
np.save("low_antiproton_ave.npy", apro_low)
np.save("runs.npy", runs)
"""

# Now for plotting and finding the bad runs.
all_pro = (pro_high, apro_high, pro_low, apro_low)
avestd = np.zeros((2, 4))
avestd[0][0] = np.mean(pro_high)
avestd[0][1] = np.mean(apro_high)
avestd[0][2] = np.mean(pro_low)
avestd[0][3] = np.mean(apro_low)
avestd[1][0] = np.std(pro_high)*3
avestd[1][1] = np.std(apro_high)*3
avestd[1][2] = np.std(pro_low)*3
avestd[1][3] = np.std(apro_low)*3

fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
labels = (r"<proton> (0.8 <= $p_T$ < 2.0)", r"<antiproton> (0.8 <= $p_T$ < 2.0)", r"<proton> (0.4 < $p_T$ < 0.8)",
          r"<antiproton> (0.4 < $p_T$ < 0.8)")

for i in range(2):
    for j in range(2):
        x = i*2 + j
        ax[i, j].plot(all_pro[x], lw=0, marker='*', c='black', mfc='r', label="average per run")
        ax[i, j].axhline(avestd[0][x], c='r', ls='--', label='mean')
        ax[i, j].axhline(avestd[0][x]+avestd[1][x], c='blue', ls='--', label=r'$3\sigma$')
        ax[i, j].axhline(avestd[0][x]-avestd[1][x], c='blue', ls='--')
        ax[i, j].set_title(labels[x])
        ax[i, j].set_ylim(0, 0.5 + (x+1 % 2)*5)
        ax[i, j].legend()
plt.show()

index = ((pro_high > avestd[0][0]+avestd[1][0]) | (apro_high > avestd[0][1]+avestd[1][1]) |
         (pro_low > avestd[0][2]+avestd[1][2]) | (apro_low > avestd[0][3]+avestd[1][3]) |
         (pro_high < avestd[0][0] - avestd[1][0]) | (apro_high < avestd[0][1] - avestd[1][1]) |
         (pro_low < avestd[0][2] - avestd[1][2]) | (apro_low < avestd[0][3] - avestd[1][3]))

print(runs[index])
print(runs[-1])
np.save("badproruns.npy", runs[index])

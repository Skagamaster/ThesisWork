import numpy as np
import uproot as up
import os
import matplotlib.pyplot as plt

'''
Events before QA:
200 GeV: 11.74M
9.2 GeV: 232k
27 GeV: 605k
19.6 GeV: 4.5M
'''

os.chdir(r'F:\2020Picos\9.2_GeV')
folders = os.listdir()
events = 0

list_pico = os.listdir()
count = 1
for j in list_pico:
    if count % 50 == 0:
        print(count, "of", len(list_pico))
    count += 1
    try:
        events += len(up.open(j)['PicoDst']['Event']['Event.mRunId'].array())
    except Exception as e:
        print("Exception on", j)
print(events)
plt.plot(0)
plt.show()

for i in folders:
    try:
        os.chdir(i)
        list_pico = os.listdir()
        count = 1
        for j in list_pico:
            if count % 50 == 0:
                print(count, "of", len(list_pico))
            count += 1
            try:
                events += len(up.open(j)['PicoDst']['Event']['Event.mRunId'].array())
            except Exception as e:
                print("Exception on", j)
        os.chdir(r'F:\2019Picos\AuAu200')
    except Exception as e:
        print("Nopers on", i)
print(events)

plt.plot(0)
plt.show()


def get_avestd(ar):
    counts = ar[0]
    bins = ar[1][:-1]
    r_sum = np.sum(counts)
    r_ave = np.sum(np.multiply(counts, bins)) / r_sum
    r_std = np.sqrt(np.sum(np.multiply(counts, np.power(np.subtract(bins, r_ave), 2))) / r_sum)
    r_std = r_std / np.sqrt(r_sum)
    return r_ave, r_std


os.chdir(r'C:\Users\dansk\Documents\Thesis\Protons\WIP\FromYu\Test_Results\BigList')
runs = []
files = os.listdir()
for i in files:
    if i.startswith('out_') & (i.endswith('.root')):
        runs.append(i)
metrics = ['refmult', 'v_z', 'p_t', 'phi', 'eta', 'dca']
ave = []
std = []
for i in range(len(metrics)):
    ave.append([])
    std.append([])
char1 = "_"
char2 = "."
count = 0
for i in runs:
    arr = up.open(i)
    num = i[i.find(char1) + 1:i.find(char2)]
    if num == '1':
        for j in range(len(metrics)):
            ave_arr = arr[metrics[j]].to_numpy()
            arr_ave, arr_std = get_avestd(ave_arr)
            ave[j].append(arr_ave)
            std[j].append(arr_std)
    else:
        for j in range(len(metrics)):
            ave_arr = arr[metrics[j] + "_{}".format(num)].to_numpy()
            arr_ave, arr_std = get_avestd(ave_arr)
            ave[j].append(arr_ave)
            std[j].append(arr_std)
    if count == 2000:
        print("breakbreakbreakbreakbreakbreakbreak")
        break
    count += 1
ave = np.asarray(ave)
std = np.array(std)
tot_ave = []
tot_std = []
badruns = []
badargs = []
for i in range(len(metrics)):
    badargs.append([])
    tot_ave.append(np.mean(ave[i]))
    tot_std.append(np.sqrt(np.sum(np.power(std[i], 2)) / len(std[i])) * 3)
    for j in range(len(ave[i])):
        # if (ave[i]+std[i] > tot_ave+tot_std) | (ave[i]-std[i] < tot_ave-tot_std):
        if (ave[i][j] + std[i][j] < tot_ave[i] - tot_std[i]) | \
                (ave[i][j] - std[i][j] > tot_ave[i] + tot_std[i]):
            badruns.append(runs[j])
            badargs[i].append(j)
    print("Bad runs in " + metrics[i] + ": " + str(len(badargs[i])))
arg_list = []
for i in range(len(badargs[0])):
    arg_list.append(runs[badargs[0][i]])
arg_list = np.asarray(arg_list, dtype='object')
print(str(np.round(100*(1-len(badargs[0])/len(runs)), 2)) + "% good runs in RefMult3.")
np.save('badruns_new.npy', arg_list)
fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
ylabels = ['<RefMult3>', r'<$v_z$>', r'<$p_T$>', r'<$\phi$>', r'<$\eta$>', '<DCA>']
for i in range(2):
    for j in range(3):
        x = i * 3 + j
        q = x
        ax[i, j].errorbar(np.linspace(0, len(ave[x]) - 1, len(ave[x])), ave[x], yerr=std[x],
                          lw=0, capsize=2, elinewidth=1, color='black')
        ax[i, j].errorbar(np.linspace(0, len(ave[x]) - 1, len(ave[x])), ave[x], yerr=std[x],
                          lw=0, capsize=2, elinewidth=1, color='black')
        ax[i, j].errorbar(np.linspace(0, len(ave[x]) - 1, len(ave[x]))[badargs[q]],
                          ave[x][badargs[q]],
                          yerr=std[x][badargs[q]], lw=0, capsize=2, elinewidth=1, color='r')
        ax[i, j].set_ylabel(ylabels[x], loc='top', fontsize=15)
        ax[i, j].set_xlabel("Run ID", loc='right', fontsize=15)
        ax[i, j].axhline(tot_ave[x], linestyle='--', c='g', label=r'$\mu$')
        ax[i, j].axhline(tot_ave[x] + tot_std[x], linestyle='--', c='b', label=r'$3\sigma$')
        ax[i, j].axhline(tot_ave[x] - tot_std[x], linestyle='--', c='b')
        ax[i, j].set_ylim(tot_ave[x] - 10*tot_std[x], tot_ave[x] + 10*tot_std[x])
        ax[i, j].legend()
plt.show()

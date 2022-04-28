import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(r"C:\Users\dansk\Documents\Thesis\Tristan")

Maxr = 1.0015801574063241

N_cp = np.load(r"N_cp_loop.npy", allow_pickle=True)
Ncoll = np.load(r"Ncoll_loop.npy", allow_pickle=True)
Npart = np.load(r"Npart_loop.npy", allow_pickle=True)
Nucleus = np.load(r"Nucleus_test.npy", allow_pickle=True)
builder = np.load(r"Ncoll_builder.npy", allow_pickle=True)
builder1 = np.load(r"Ncoll_builder1.npy", allow_pickle=True)
b = np.load(r"b_loop.npy", allow_pickle=True)
a = np.where(Ncoll == np.max(Ncoll))[0][0]
a = 1
print(a)
nuctile1 = np.tile(Nucleus[a][0], (197, 1)).T
nuctile2 = np.tile(Nucleus[a][1], (197, 1))
nuctile3 = np.tile(Nucleus[a][3], (197, 1))
nuctile4 = np.tile(Nucleus[a][2], (197, 1)).T
Ncoll_builder = np.sqrt(np.add(np.power(np.subtract(nuctile1, nuctile2), 2),
                                np.power(np.add(b[a]-nuctile4, nuctile3), 2))).T
Maxr = 1.0015801574063241

plt.hist(np.hstack(builder), histtype='step', bins=100, alpha=0.5)
plt.hist(np.hstack(builder1), histtype='step', bins=100, alpha=0.5)
plt.show()
print(len(np.hstack(builder)), len(np.hstack(builder1)))


print(Ncoll)
testarr = np.where(N_cp[a] <= Maxr)
print(N_cp[a] <= Maxr)
print(np.sum(Ncoll_builder <= Maxr))

plt.hist(np.hstack(Ncoll_builder), bins=300, histtype='step', label="Python", alpha=0.5)
plt.hist(np.hstack(N_cp[a]), bins=300, histtype='step', label="Stupid Loop", alpha=0.5)
plt.legend()
plt.show()

plt.plot(N_cp[a][0], Ncoll_builder[0])
plt.show()


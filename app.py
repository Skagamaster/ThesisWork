import numpy as np
import uproot as up
import os
import matplotlib.pyplot as plt

centrality_measure = (r"$X_{RM3}$", r"$X_{ReLU}$")
cumulants = (r'$C_1$', r'$C_2$', r'$C_3$', r'$C_4$')
cumulantsr = (r'$\frac{C_2}{C_1}$', r'$\frac{C_3}{C_2}$')

rm3_c1 = np.array((1.97368669, 3.44347222, 5.61693877, 8.47709283, 12.29781965, 17.45065261,
                  22.60512095, 26.98056633))
rm3_e1 = np.array((0.0121029, 0.01700739, 0.02369137, 0.03141459, 0.04046229, 0.05064509,
                  0.05991962, 0.07850044))

relu_c1 = np.array((1.69004985, 3.02473652, 5.08824115, 8.03173287, 12.16937418, 17.78914778,
                   23.18334089, 27.71983568))
relu_e1 = np.array((0.01108431, 0.01575673, 0.02175224, 0.02866057, 0.0375144, 0.04894231,
                   0.06136381, 0.08694808))

rm3_c2 = np.array((3.42493589, 5.76306646, 9.36163177, 13.68836177, 18.93224586, 24.94045069,
                  30.75880902, 33.65132598))
rm3_e2 = np.array((0.02861559, 0.0390038, 0.05615742, 0.07693027, 0.10424427, 0.13723564,
                  0.17094026, 0.22547745))

relu_c2 = np.array((2.79761642, 4.44565264, 6.91104341, 10.22678832, 14.87195751, 20.90279583,
                   26.5045823, 31.2526008))
relu_e2 = np.array((0.02539398, 0.03417815, 0.04837238, 0.0663301, 0.09192418, 0.12764322,
                   0.16979523, 0.24808371))

rm3_c3 = np.array((10.7421606, 15.06310652, 22.20409447, 27.8267386, 33.31289114, 37.32115172,
                  42.99470012, 34.47701289))
rm3_e3 = np.array((1.33574788, 1.68367739, 2.66006594, 3.58215982, 5.54656182, 8.28997563,
                  12.61752879, 16.57936278))

relu_c3 = np.array((8.32795083, 10.17302592, 13.44857261, 17.25079253, 21.863122, 24.60704712,
                   26.82273252, 25.10168947))
relu_e3 = np.array((0.96402933, 1.11284449, 1.81838997, 2.6395478, 4.13553695, 6.8070667,
                   11.1772611, 18.01751313))

relu_c = (relu_c1, relu_c2, relu_c3)
relu_e = (relu_e1, relu_e2, relu_e3)
rm3_c = (rm3_c1, rm3_c2, rm3_c3)
rm3_e = (rm3_e1, rm3_e2, rm3_e3)

x_cent = ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%',
          '40-50%', '30-40%', '20-30%', '10-20%', '5-10%', '0-5%']
x = np.linspace(0, 10, 11)
x_offset = np.linspace(0.3, 10.3, 11)
ms = 10
ew = 1
c1 = 'r'
c2 = 'k'
c3 = 'purple'
c_arr = [c1, c2, c3]

fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for i in range(2):
    for j in range(2):
        r = i * 2 + j
        if r == 3:
            ax[i, j].set_axis_off()
        else:
            ax[i, j].errorbar(x_cent[3:], rm3_c[r], yerr=rm3_e[r], marker='o', color='red', ms=ms,
                              mfc='None', elinewidth=ew, lw=0, ecolor='red', label=r'$X_{ReLU}$')
            ax[i, j].errorbar(x_cent[3:], relu_c[r], yerr=relu_e[r], marker='X', color='purple', ms=ms,
                              mfc='None', elinewidth=ew, lw=0, ecolor='purple', label=r'$X_{RM3}$')
            ax[i, j].legend()
            ax[i, j].set_ylabel(cumulants[r], fontsize=20)
            ax[i, j].set_xlabel("Centrality", fontsize=20)
plt.show()
plt.close()

rm3_cr21 = np.divide(rm3_c2, rm3_c1)
relu_cr21 = np.divide(relu_c2, relu_c1)
rm3_cr32 = np.divide(rm3_c3, rm3_c2)
relu_cr32 = np.divide(relu_c3, relu_c2)

rm3_er21 = np.multiply(np.sqrt(np.add(np.power(np.divide(rm3_e1, rm3_c1), 2),
                                      np.power(np.divide(rm3_e2, rm3_c2), 2))), rm3_cr21)
relu_er21 = np.multiply(np.sqrt(np.add(np.power(np.divide(relu_e1, relu_c1), 2),
                                       np.power(np.divide(relu_e2, relu_c2), 2))), relu_cr21)
rm3_er32 = np.multiply(np.sqrt(np.add(np.power(np.divide(rm3_e3, rm3_c3), 2),
                                      np.power(np.divide(rm3_e2, rm3_c2), 2))), rm3_cr32)
relu_er32 = np.multiply(np.sqrt(np.add(np.power(np.divide(relu_e3, relu_c3), 2),
                                       np.power(np.divide(relu_e2, relu_c2), 2))), relu_cr32)

rm3_cr = (rm3_cr21, rm3_cr32)
relu_cr = (relu_cr21, relu_cr32)
rm3_er = (rm3_er21, rm3_er32)
relu_er = (relu_er21, relu_er32)

print(rm3_c[0])
print(rm3_e[0])
print(rm3_cr21)
print(rm3_e[0])
print(rm3_e[1])
print(rm3_er21)

fig, ax = plt.subplots(2, figsize=(8, 8), constrained_layout=True)
for i in range(2):
    ax[i].errorbar(x_cent[3:], rm3_cr[i], yerr=rm3_er[i], marker='o', color='red', ms=ms,
                      mfc='None', elinewidth=ew, lw=0, ecolor='red', label=r'$X_{ReLU}$')
    ax[i].errorbar(x_cent[3:], relu_cr[i], yerr=relu_er[i], marker='X', color='purple', ms=ms,
                      mfc='None', elinewidth=ew, lw=0, ecolor='purple', label=r'$X_{RM3}$')
    ax[i].legend()
    ax[i].set_ylabel(cumulantsr[i], fontsize=20)
    ax[i].set_xlabel("Centrality", fontsize=20)
plt.show()
plt.close()

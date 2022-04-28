import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages

n_vals = np.linspace(2, 3.5, 10)
p_vals = np.linspace(0.6, 0.9, 10)
a_vals = np.linspace(0.4, 1.0, 10)
pred_loc = r"C:\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits\9_2021\\"
data = np.load(pred_loc+"pearsons.npy", allow_pickle=True)
names = ["RefMult3", r"$\Sigma$ EPD", r"$\Sigma EPD_{out}$", r"$X_{\zeta, LW}$",
         r"$X_{\zeta, RELU}$", r"$X_{\zeta, swish}$", r"$X_{\zeta, mish}$"]
print("Starting our attack run.")
with PdfPages(r'\Users\dansk\Documents\Thesis\2021_DNP_Work\ML_fits\9_2021\Pearson_Plots.pdf') as export_pdf:
    for i in range(len(data)):
        print("Rogue", i, "standing by.")
        X, Y = np.meshgrid(p_vals, a_vals)
        fig, ax = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
        fig.suptitle(names[i], fontsize=20)
        for m in range(3):
            for n in range(4):
                x = m*4 + n
                if x >= 10:
                    continue
                im = ax[m, n].pcolormesh(X, Y, data[i][x], norm=LogNorm(), cmap='jet', shading='auto')
                ax[m, n].set_title("n= {}".format(n_vals[x]), fontsize=20)
                ax[m, n].set_xlabel("p", fontsize=15)
                ax[m, n].set_ylabel(r"$\alpha$", fontsize=15)
                fig.colorbar(im, ax=ax[m, n])
        ax[2, 2].set_axis_off()
        ax[2, 3].set_axis_off()
        export_pdf.savefig()
        plt.close()
print("You've switched off your targeting computer. What's wrong?")

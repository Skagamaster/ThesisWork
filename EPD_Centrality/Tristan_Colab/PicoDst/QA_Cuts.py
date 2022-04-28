import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# If there is data already saved, then set this to True. If not, set it to False.
already_saved = True

# Use this if there is no saved data already to add to.
if already_saved is not True:
    print("Generating new data set.")
    # Averages over all runs.
    averages = pd.DataFrame(columns=['ave_refmult3', 'err_refmults', 'ave_vz', 'err_vz', 'ave_vr', 'err_vr',
                                     'ave_pt', 'err_pt', 'ave_eta', 'err_eta', 'ave_zdcx', 'err_zdcx',
                                     'ave_phi', 'err_phi', 'ave_dca', 'err_dca'])

    # Arrays to hold our histogram data for before and after QA cut analysis.
    a, b, c, d = 1000, 161, 86, 101
    vz_count, vz_bins = np.histogram(0, bins=a, range=(-200, 200))
    vr_count, vr_binsX, vr_binsY = np.histogram2d([0], [0], bins=a, range=((-10, 10), (-10, 10)))
    ref_count, ref_bins = np.histogram(0, bins=a, range=(0, a))
    mpq_count, mpq_binsX, mpq_binsY = np.histogram2d([0], [0], bins=a, range=((0, 1.5), (-5, 5)))
    rt_mult_count, rt_mult_binsX, rt_mult_binsY = np.histogram2d([0], [0], bins=(1700, a),
                                                                 range=((0, 1700), (0, 1000)))
    rt_match_count, rt_match_binsX, rt_match_binsY = np.histogram2d([0], [0], bins=a, range=((0, 500), (0, 1000)))
    ref_beta_count, ref_beta_binsX, ref_beta_binsY = np.histogram2d([0], [0], bins=(400, a),
                                                                    range=((0, 400), (0, 1000)))
    pt_count, pt_bins = np.histogram(0, bins=a, range=(0, 6))
    phi_count, phi_bins = np.histogram(0, bins=a, range=(-np.pi - 0.2, np.pi + 0.2))
    dca_count, dca_bins = np.histogram(0, bins=a, range=(0, 5))
    eta_count, eta_bins = np.histogram(0, bins=a, range=(-3, 3))
    nhitsq_count, nhitsq_bins = np.histogram(0, bins=b, range=(-(b-1)/2, (b-1)/2))
    nhits_dedx_count, nhits_dedx_bins = np.histogram(0, bins=c, range=(0, c-1))
    betap_count, betap_binsX, betap_binsY = np.histogram2d([0], [0], bins=a, range=((0.5, 3.6), (0, 10)))
    dedx_pq_count, dedx_pq_binsX, dedx_pq_binsY = np.histogram2d([0], [0], bins=a, range=((0, 31), (-3, 3)))
    runs = np.empty(0)


# Use this if you already have saved data.
if already_saved is True:
    print("Building from existing data set.")
    os.chdir(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_4_14_2021\totals")
    averages = pd.read_pickle("averages.pkl")
    vz_count = np.load("vz_hist.npy", allow_pickle=True)[0]
    vz_bins = np.load("vz_hist.npy", allow_pickle=True)[1]
    vr_count = np.load("vr_hist.npy", allow_pickle=True)[0]
    vr_binsX = np.load("vr_hist.npy", allow_pickle=True)[1]
    vr_binsY = np.load("vr_hist.npy", allow_pickle=True)[2]
    ref_count = np.load("ref_hist.npy", allow_pickle=True)[0]
    ref_bins = np.load("ref_hist.npy", allow_pickle=True)[1]
    mpq_count = np.load("mpq_hist.npy", allow_pickle=True)[0]
    mpq_binsX = np.load("mpq_hist.npy", allow_pickle=True)[1]
    mpq_binsY = np.load("mpq_hist.npy", allow_pickle=True)[2]
    rt_mult_count = np.load("rt_mult_hist.npy", allow_pickle=True)[0]
    rt_mult_binsX = np.load("rt_mult_hist.npy", allow_pickle=True)[1]
    rt_mult_binsY = np.load("rt_mult_hist.npy", allow_pickle=True)[2]
    rt_match_count = np.load("rt_match_hist.npy", allow_pickle=True)[0]
    rt_match_binsX = np.load("rt_match_hist.npy", allow_pickle=True)[1]
    rt_match_binsY = np.load("rt_match_hist.npy", allow_pickle=True)[2]
    ref_beta_count = np.load("ref_beta_hist.npy", allow_pickle=True)[0]
    ref_beta_binsX = np.load("ref_beta_hist.npy", allow_pickle=True)[1]
    ref_beta_binsY = np.load("ref_beta_hist.npy", allow_pickle=True)[2]
    pt_count = np.load("pt_hist.npy", allow_pickle=True)[0]
    pt_bins = np.load("pt_hist.npy", allow_pickle=True)[1]
    phi_count = np.load("phi_hist.npy", allow_pickle=True)[0]
    phi_bins = np.load("phi_hist.npy", allow_pickle=True)[1]
    dca_count = np.load("dca_hist.npy", allow_pickle=True)[0]
    dca_bins = np.load("dca_hist.npy", allow_pickle=True)[1]
    eta_count = np.load("eta_hist.npy", allow_pickle=True)[0]
    eta_bins = np.load("eta_hist.npy", allow_pickle=True)[1]
    nhitsq_count = np.load("nhitsq_hist.npy", allow_pickle=True)[0]
    nhitsq_bins = np.load("nhitsq_hist.npy", allow_pickle=True)[1]
    nhits_dedx_count = np.load("nhits_dedx_hist.npy", allow_pickle=True)[0]
    nhits_dedx_bins = np.load("nhits_dedx_hist.npy", allow_pickle=True)[1]
    betap_count = np.load("betap_hist.npy", allow_pickle=True)[0]
    betap_binsX = np.load("betap_hist.npy", allow_pickle=True)[1]
    betap_binsY = np.load("betap_hist.npy", allow_pickle=True)[2]
    dedx_pq_count = np.load("dedx_pq_hist.npy", allow_pickle=True)[0]
    dedx_pq_binsX = np.load("dedx_pq_hist.npy", allow_pickle=True)[1]
    dedx_pq_binsY = np.load("dedx_pq_hist.npy", allow_pickle=True)[2]
    runs = np.load("runs.npy", allow_pickle=True)
print(runs)
print("Loading new data.")
os.chdir(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_4_14_2021")
for file in os.listdir():
    if "ave" in file:
        df = pd.read_pickle(file)
        frames = [averages, df]
        averages = pd.concat(frames)
        averages.reset_index()
    elif "vz" in file:
        vz_count += np.load(file, allow_pickle=True)[0]
    elif "ref_hist" in file:
        ref_count += np.load(file, allow_pickle=True)[0]
    elif "mpq" in file:
        mpq_count += np.load(file, allow_pickle=True)[0]
    elif "rt_mult" in file:
        rt_mult_count += np.load(file, allow_pickle=True)[0]
    elif "rt_match" in file:
        rt_match_count += np.load(file, allow_pickle=True)[0]
    elif "ref_beta" in file:
        ref_beta_count += np.load(file, allow_pickle=True)[0]
    elif "pt_hist" in file:
        pt_count += np.load(file, allow_pickle=True)[0]
    elif "phi" in file:
        phi_count += np.load(file, allow_pickle=True)[0]
    elif "dca" in file:
        dca_count += np.load(file, allow_pickle=True)[0]
    elif "eta_hist" in file:
        eta_count += np.load(file, allow_pickle=True)[0]
    elif "nhitsq" in file:
        nhitsq_count += np.load(file, allow_pickle=True)[0]
    elif "nhits_dedx" in file:
        nhits_dedx_count += np.load(file, allow_pickle=True)[0]
    elif "betap" in file:
        betap_count += np.load(file, allow_pickle=True)[0]
    elif "dedx_pq" in file:
        dedx_pq_count += np.load(file, allow_pickle=True)[0]
    elif "runs" in file:
        runner = np.load(file, allow_pickle=True)

runs = np.hstack((runs, runner))
print(runs)
print("Data loaded; plotting averages.")

ave_len = len(averages)
ave_len = np.linspace(0, ave_len-1, ave_len)
columns = ['ave_refmult3', 'err_refmults', 'ave_vz', 'err_vz', 'ave_vr', 'err_vr', 'ave_pt', 'err_pt', 'ave_eta',
           'err_eta', 'ave_zdcx', 'err_zdcx', 'ave_phi', 'err_phi', 'ave_dca', 'err_dca']
ave_titles = [r"$RefMult3$", r"$v_z$", r"$v_r$", r"$p_T$", r"$\eta$", r"$ZDC_x$", r"$\phi$", r"$DCA$"]
ave_ylabels = [r"<$RefMult3$>", r"<$v_z$> (cm)", r"<$v_r$> (cm)", r"<$p_T> (\frac{GeV}{c})$",
               r"<$\eta$>", r"<$ZDC_x$> (cm)", r"<$\phi$> (rad)", r"<$DCA$> (cm)"]

# Let's make some plots!
fig, ax = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True)
for i in range(2):
    for j in range(4):
        x = i*4 + j
        ave = np.mean(averages[columns[x*2]])
        std = 3*np.std(averages[columns[x*2]])
        ax[i, j].errorbar(ave_len, averages[columns[2*x]].to_numpy(), yerr=averages[columns[2*x + 1]].to_numpy(),
                          fmt='ok', ms=0, mfc='None', capsize=1.5, elinewidth=1)
        ax[i, j].set_xticks(ax[0, 0].get_xticks()[::100])
        ax[i, j].tick_params(labelrotation=45, labelsize=7)
        ax[i, j].axhline(ave, 0, 1, c='red', ls="--", label="Mean")
        ax[i, j].axhline(ave+std, 0, 1, c='blue', ls="--", label=r'3$\sigma$')
        ax[i, j].axhline(ave-std, 0, 1, c='blue', ls="--")
        ax[i, j].set_title(ave_titles[x], fontsize=20)
        ax[i, j].set_ylabel(ave_ylabels[x], fontsize=10)
        ax[i, j].legend()
plt.show()

X, Y = np.meshgrid(ref_beta_binsY, ref_beta_binsX)
plt.pcolormesh(X, Y, ref_beta_count, cmap="jet", norm=colors.LogNorm())
plt.xlabel("RefMult3", fontsize=10)
plt.ylabel(r"$\beta \eta$1", fontsize=10)
plt.title(r"RefMult3 vs $\beta \eta$1", fontsize=20)
plt.ylim(0, np.max(Y))
plt.colorbar()
plt.show()

print("Saving updated data.")
os.chdir(r"C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_4_14_2021\totals")

averages.to_pickle("averages.pkl")
np.save("vz_hist.npy", (vz_count, vz_bins))
np.save("vr_hist.npy", (vr_count, vr_binsX, vr_binsY))
np.save("ref_hist.npy", (ref_count, ref_bins))
np.save("mpq_hist.npy", (mpq_count, mpq_binsX, mpq_binsY))
np.save("rt_mult_hist.npy", (rt_mult_count, rt_mult_binsX, rt_mult_binsY))
np.save("rt_match_hist.npy", (rt_match_count, rt_match_binsX, rt_match_binsY))
np.save("ref_beta_hist.npy", (ref_beta_count, ref_beta_binsX, ref_beta_binsY))
np.save("pt_hist.npy", (pt_count, pt_bins))
np.save("phi_hist.npy", (phi_count, phi_bins))
np.save("dca_hist.npy", (dca_count, dca_bins))
np.save("eta_hist.npy", (eta_count, eta_bins))
np.save("nhitsq_hist.npy", (nhitsq_count, nhitsq_bins))
np.save("nhits_dedx_hist.npy", (nhits_dedx_count, nhits_dedx_bins))
np.save("betap_hist.npy", (betap_count, betap_binsX, betap_binsY))
np.save("dedx_pq_hist.npy", (dedx_pq_count, dedx_pq_binsX, dedx_pq_binsY))
np.save("runs.npy", runs)

directory = r"C:\Users\dansk\Documents\Thesis\Protons\WIP\QA_4_14_2021"

print("Deleting old data.")
test = os.listdir(directory)
for item in test:
    if item.endswith(".npy"):
        os.remove(os.path.join(directory, item))
    if item.endswith(".pkl"):
        os.remove(os.path.join(directory, item))
print("The last run was " + str(runs[-1]) + ".")

